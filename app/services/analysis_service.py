"""
analysis_service.py
===================
Business logic layer — bridges NBA data retrieval with Claude-powered analysis.

Responsibilities
----------------
- Prompt construction from domain data
- Stat aggregation and trend computation
- Routing to the correct Claude system prompt per analysis type
- Response shaping for API consumers

This module contains no HTTP code (that lives in nba_service.py) and no
Anthropic SDK calls (those live in claude_service.py).
"""

from __future__ import annotations

import datetime
import logging
from typing import Any, Optional

from app.models.schemas import Game, GameAnalysisResponse
from app.services import claude_service, nba_service

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_SEASON: int = 2025
_RECENT_FORM_WINDOW: int = 10   # number of most-recent games used for trend data
_MAX_TRADE_PLAYERS: int = 4     # max players to fetch live stats for in a trade


# ---------------------------------------------------------------------------
# System Prompts
# ---------------------------------------------------------------------------

NBA_ANALYST_SYSTEM_PROMPT: str = """You are the highest-paid NBA analyst in the country. Your clients are GMs, bettors, and executives who pay premium money for your edge. They don't want summaries. They want what you actually think.

You have watched more NBA film than anyone in this conversation. You know pace differentials, defensive rating trends, how teams perform on back-to-backs, which coaches make in-game adjustments and which ones don't, which stars disappear in fourth quarters. You use that knowledge.

When analyzing a game: open with the sharpest thing you know about this matchup — the thing most people miss. Then cover the stylistic clash, the one player who will determine the outcome, and the specific reason one team wins. Close with a confident, unhedged prediction.

When analyzing a player: open with what the numbers actually mean in context — not just what they are. Call out if a player is overrated, underrated, declining, or ascending. Reference the last 10 games trend vs season average and explain what it signals. End with one sentence that captures exactly where this player stands right now.

Every word must earn its place. If a sentence doesn't add information or edge, cut it. No throat-clearing. No "it's worth noting." No "at the end of the day." Start with the insight, not the setup.

FORMATTING — NON-NEGOTIABLE:
Plain prose only. No markdown. No asterisks, no pound signs, no dashes used as bullets, no numbered lists, no bold, no italics, no horizontal rules, no headers. Paragraphs separated by one blank line. Write like a column in The Athletic or a Sharp report — dense, confident, readable."""


FRONT_OFFICE_SYSTEM_PROMPT: str = """You are one of the most respected front office minds in the NBA. You have built contenders, navigated the cap, and executed trades that reshaped franchises. GMs call you before they pull the trigger on anything significant.

When evaluating a trade: your first sentence is your verdict — who wins and why. Then back it up. Cover the contract values and timelines, the age curves, the fit on both ends of the floor, the second-order effects on team chemistry and depth. If a trade looks bad, say it's bad. If one side is getting fleeced, name it. Give a confidence level on your verdict.

When analyzing a roster: open with the most honest assessment of where this team actually is — not where they think they are. Identify the single biggest problem and the specific move that fixes it. Name overpaid players and call them overpaid. Identify trade candidates and explain who would actually want them and why. Give two actionable recommendations the front office should execute.

You do not hedge. You do not say "it depends." You give a position and defend it with specifics.

FORMATTING — NON-NEGOTIABLE:
Plain prose only. No markdown. No asterisks, no pound signs, no dashes used as bullets, no numbered lists, no bold, no italics, no horizontal rules, no headers. Paragraphs separated by one blank line. Dense, confident, readable prose. Front offices don't pay for maybe."""


COACH_SYSTEM_PROMPT: str = """You are an elite NBA head coach with a championship pedigree. You have full film room access, a complete coaching staff, and the live box score in front of you. Coaches and scouts pay for your input because you see things other people miss and you give answers without wasting time.

When making in-game adjustments: your first sentence identifies the single most important problem and the fix. Then go deeper — name the specific players involved, name the scheme, explain why it works against what this opponent is running right now. Use the box score data you are given. If a player has 4 fouls, that changes the lineup. If a player is 0-for-5 from three, you don't run them off screens. Use the actual numbers.

When drawing up a timeout play: name the play first. Then describe the motion in plain terms a coach could draw up in 20 seconds. Name who sets the screen, who gets the ball, what the primary read is, what the secondary read is if the first option is taken away. Close with one sentence on why this specific play works against what the defense is likely running.

You never ask for more information. You work with what you have. You give answers, not questions.

FORMATTING — NON-NEGOTIABLE:
Plain prose only. No markdown. No asterisks, no pound signs, no dashes used as bullets, no numbered lists, no bold, no italics, no horizontal rules, no headers. Write your adjustment priorities as short prose paragraphs separated by blank lines — not list items. Dense, decisive, readable. Coaches need answers in 20 seconds."""


# ---------------------------------------------------------------------------
# Section-Specific Analyst Prompts
# ---------------------------------------------------------------------------

SECTION_PROMPTS: dict[str, str] = {
    "offense": (
        "Give a deep offensive breakdown of this player. Cover scoring volume, shot selection, "
        "efficiency by zone, usage rate, creation vs off-ball role, pick and roll tendencies, "
        "and how defenses are currently scheming against them. Be specific with the numbers."
    ),
    "defense": (
        "Give a deep defensive breakdown. Cover on-ball defense, help defense, switching ability, "
        "defensive rebounding, steal and block tendencies, the matchup problems they create or solve, "
        "and their measurable impact on team defensive rating."
    ),
    "off_the_court": (
        "Analyze this player's off-court impact: leadership, locker room presence, media profile, "
        "coachability, history of team chemistry effects, and what kind of environment they thrive "
        "in vs struggle in."
    ),
    "injuries": (
        "Analyze this player's injury history and durability. Cover games-played trends over the "
        "last 3 seasons, known injury history, load management patterns, position-related injury "
        "risk at their age, and what to watch for going forward."
    ),
    "financials": (
        "Break down this player's financial situation: estimated contract value vs production, "
        "whether they are an overpaid or underpaid asset, trade value relative to salary, years "
        "remaining context, and what a front office should think about this contract."
    ),
}


# ---------------------------------------------------------------------------
# Shared Stat Utilities
# ---------------------------------------------------------------------------

def _safe_avg(values: list[Optional[float]]) -> float:
    """
    Compute the arithmetic mean of a list, ignoring ``None`` entries.

    Returns 0.0 when the list is empty or contains only ``None`` values,
    rather than raising ``ZeroDivisionError``.

    Parameters
    ----------
    values:
        List of numeric values, potentially containing ``None``.

    Returns
    -------
    float
        Mean of non-null values, rounded to 1 decimal place.
    """
    clean: list[float] = [v for v in values if v is not None]
    return round(sum(clean) / len(clean), 1) if clean else 0.0


def _trend_label(recent: float, season: float) -> str:
    """
    Return a signed difference string showing recent-form vs season average.

    Examples: ``"+3.2"``, ``"-1.5"``, ``"+0.0"``

    Parameters
    ----------
    recent:
        Average over the most recent N games.
    season:
        Full-season average.

    Returns
    -------
    str
        Signed delta, e.g. ``"+2.1"`` or ``"-0.8"``.
    """
    diff = round(recent - season, 1)
    sign = "+" if diff >= 0 else ""
    return f"{sign}{diff}"


def _pct_trend_label(recent: float, season: float) -> str:
    """
    Variant of ``_trend_label`` for percentage values stored as decimals.

    Multiplies both values by 100 before computing the delta so the output
    reads as a percentage-point difference, e.g. ``"+4.2%"`` rather than
    ``"+0.042"``.

    Parameters
    ----------
    recent:
        Recent field-goal (or other) percentage as a decimal (0.0–1.0).
    season:
        Season field-goal (or other) percentage as a decimal (0.0–1.0).

    Returns
    -------
    str
        Signed percentage-point delta, e.g. ``"+4.2%"``.
    """
    diff = round((recent - season) * 100, 1)
    sign = "+" if diff >= 0 else ""
    return f"{sign}{diff}%"


# ---------------------------------------------------------------------------
# Game Formatting Helpers
# ---------------------------------------------------------------------------

def _format_single_game(game: Game) -> str:
    """
    Render a single ``Game`` as a multi-line display string for prompt injection.

    Shows the score if either team has scored, otherwise shows the matchup in
    the conventional away @ home format.

    Parameters
    ----------
    game:
        Hydrated ``Game`` domain object.

    Returns
    -------
    str
        Three-line string: score line, home team detail, away team detail.
    """
    home = game.home_team
    away = game.visitor_team
    has_score = game.home_team_score > 0 or game.visitor_team_score > 0

    score_line = (
        f"{away.abbreviation} {game.visitor_team_score} — "
        f"{game.home_team_score} {home.abbreviation}"
        if has_score
        else f"{away.abbreviation} @ {home.abbreviation}"
    )

    return (
        f"{score_line} | {game.status}\n"
        f"  HOME: {home.city} {home.name} ({home.conference} / {home.division})\n"
        f"  AWAY: {away.city} {away.name} ({away.conference} / {away.division})"
    )


def _format_games_for_prompt(games: list[Game]) -> str:
    """
    Render all games in a date's slate as a single prompt-ready block.

    Parameters
    ----------
    games:
        List of ``Game`` objects for a given date.

    Returns
    -------
    str
        Newline-separated game display strings, or a no-games notice.
    """
    if not games:
        return "No games scheduled for this date."
    return "\n\n".join(_format_single_game(g) for g in games)


# ---------------------------------------------------------------------------
# Player Stat Aggregation
# ---------------------------------------------------------------------------

async def _build_player_stat_block(player_name: str, season: int) -> tuple[Any, dict[str, Any]]:
    """
    Look up a player, fetch their game logs and official averages, and return
    both the ``Player`` object and a dict of computed stat aggregates.

    This helper is shared by ``analyze_player`` and ``analyze_player_section``
    to eliminate the stat-aggregation code duplication that existed in the
    original implementation.

    Parameters
    ----------
    player_name:
        Full or partial player name. We search by the full name string and fall
        back to the last token only if no results are returned.
    season:
        NBA season start year.

    Returns
    -------
    tuple[Player, dict]
        ``(player, aggregates)`` where ``aggregates`` contains keys:
        ``total_games``, ``avg_pts``, ``avg_reb``, ``avg_ast``, ``avg_stl``,
        ``avg_blk``, ``avg_fg``, ``avg_fg3``, ``avg_ft``, ``recent_pts``,
        ``recent_reb``, ``recent_ast``, ``recent_stl``, ``recent_blk``,
        ``recent_fg``, ``recent_fg3``.

    Raises
    ------
    ValueError
        If no player matching ``player_name`` can be found.
    """
    clean_name = player_name.strip()

    # Try the full name first; fall back to the last token (surname) only if
    # the full-name search returns nothing. The original code always used only
    # the last token which caused incorrect matches for players with short,
    # common surnames (e.g. "Grant" matching the wrong player).
    players = await nba_service.search_players(clean_name)

    if not players and " " in clean_name:
        last_token = clean_name.split()[-1]
        logger.debug(
            "Full-name search for %r returned no results; retrying with last token %r",
            clean_name,
            last_token,
        )
        players = await nba_service.search_players(last_token)

    if not players:
        raise ValueError(f"No player found matching '{player_name}'")

    player = players[0]
    logger.info(
        "Player resolved | query=%r matched=%s %s id=%d",
        player_name,
        player.first_name,
        player.last_name,
        player.id,
    )

    stats = await nba_service.get_player_stats(player.id, season)
    official = await nba_service.get_season_averages(player.id, season)

    total_games = len(stats)

    # Full-season averages: prefer the official endpoint values; fall back to
    # computing from raw game logs when the season-averages endpoint returns
    # nothing (common mid-season or for players with limited appearances).
    avg_pts = round(float(official.get("pts") or _safe_avg([s.points for s in stats])), 1)
    avg_reb = round(float(official.get("reb") or _safe_avg([s.rebounds for s in stats])), 1)
    avg_ast = round(float(official.get("ast") or _safe_avg([s.assists for s in stats])), 1)
    avg_stl = round(float(official.get("stl") or _safe_avg([s.steals for s in stats])), 1)
    avg_blk = round(float(official.get("blk") or _safe_avg([s.blocks for s in stats])), 1)
    avg_fg  = round(float(official.get("fg_pct") or _safe_avg([s.fg_pct for s in stats])), 3)
    avg_fg3 = round(float(official.get("fg3_pct") or _safe_avg([s.fg3_pct for s in stats])), 3)
    avg_ft  = round(float(official.get("ft_pct") or _safe_avg([s.ft_pct for s in stats])), 3)

    # Recent form: last N games by chronological order (stats are pre-sorted
    # ascending by game_id in nba_service.get_player_stats).
    recent = stats[-_RECENT_FORM_WINDOW:]

    recent_pts  = _safe_avg([s.points for s in recent])
    recent_reb  = _safe_avg([s.rebounds for s in recent])
    recent_ast  = _safe_avg([s.assists for s in recent])
    recent_stl  = _safe_avg([s.steals for s in recent])
    recent_blk  = _safe_avg([s.blocks for s in recent])
    recent_fg   = _safe_avg([s.fg_pct for s in recent])
    recent_fg3  = _safe_avg([s.fg3_pct for s in recent])

    aggregates: dict[str, Any] = {
        "total_games": total_games,
        "avg_pts": avg_pts,
        "avg_reb": avg_reb,
        "avg_ast": avg_ast,
        "avg_stl": avg_stl,
        "avg_blk": avg_blk,
        "avg_fg": avg_fg,
        "avg_fg3": avg_fg3,
        "avg_ft": avg_ft,
        "recent_pts": recent_pts,
        "recent_reb": recent_reb,
        "recent_ast": recent_ast,
        "recent_stl": recent_stl,
        "recent_blk": recent_blk,
        "recent_fg": recent_fg,
        "recent_fg3": recent_fg3,
    }

    return player, aggregates


def _render_stat_block(player: Any, season: int, agg: dict[str, Any]) -> str:
    """
    Format a fully assembled stat block string for injection into a Claude prompt.

    Parameters
    ----------
    player:
        ``Player`` domain object.
    season:
        NBA season start year.
    agg:
        Aggregates dict produced by ``_build_player_stat_block``.

    Returns
    -------
    str
        Multi-line stat block, plain text, no markdown.
    """
    team_name = player.team.name if player.team else "N/A"

    return (
        f"Player: {player.first_name} {player.last_name}\n"
        f"Team: {team_name} | Position: {player.position or 'N/A'} | "
        f"Season: {season} | Games: {agg['total_games']}\n"
        f"\n"
        f"SEASON AVERAGES:\n"
        f"  PTS: {agg['avg_pts']} | REB: {agg['avg_reb']} | AST: {agg['avg_ast']} | "
        f"STL: {agg['avg_stl']} | BLK: {agg['avg_blk']}\n"
        f"  FG%: {agg['avg_fg']:.1%} | 3P%: {agg['avg_fg3']:.1%} | FT%: {agg['avg_ft']:.1%}\n"
        f"\n"
        f"LAST {_RECENT_FORM_WINDOW} GAMES:\n"
        f"  PTS: {agg['recent_pts']} ({_trend_label(agg['recent_pts'], agg['avg_pts'])} vs season) | "
        f"REB: {agg['recent_reb']} ({_trend_label(agg['recent_reb'], agg['avg_reb'])}) | "
        f"AST: {agg['recent_ast']} ({_trend_label(agg['recent_ast'], agg['avg_ast'])})\n"
        f"  STL: {agg['recent_stl']} ({_trend_label(agg['recent_stl'], agg['avg_stl'])}) | "
        f"BLK: {agg['recent_blk']} ({_trend_label(agg['recent_blk'], agg['avg_blk'])})\n"
        f"  FG%: {agg['recent_fg']:.1%} ({_pct_trend_label(agg['recent_fg'], agg['avg_fg'])}) | "
        f"3P%: {agg['recent_fg3']:.1%} ({_pct_trend_label(agg['recent_fg3'], agg['avg_fg3'])})"
    )


# ---------------------------------------------------------------------------
# Public Analysis Functions
# ---------------------------------------------------------------------------

async def analyze_today_games(target_date: Optional[str] = None) -> GameAnalysisResponse:
    """
    Fetch the NBA slate for a date and return a Claude-generated analyst breakdown.

    Parameters
    ----------
    target_date:
        ISO-8601 date string (``"YYYY-MM-DD"``). Defaults to today in US
        Central Time when omitted.

    Returns
    -------
    GameAnalysisResponse
        Contains the game list, full analysis text, model metadata, and token usage.
    """
    date_label = target_date or "today"
    logger.info("Analyzing game slate | date=%s", date_label)

    games = await nba_service.get_games_by_date(target_date)
    game_summary = _format_games_for_prompt(games)

    prompt = (
        f"NBA slate for {date_label} — {len(games)} game(s):\n\n"
        f"{game_summary}\n\n"
        f"Give a full analyst breakdown of tonight's slate. For each game: identify the key "
        f"matchup edge, the style-of-play clash, and make a prediction. Close with your best "
        f"game of the night and why."
    )

    result = await claude_service.analyze(
        prompt=prompt,
        system_prompt=NBA_ANALYST_SYSTEM_PROMPT,
    )

    logger.info(
        "Game slate analysis complete | games=%d tokens=%d",
        len(games),
        result.tokens_used,
    )

    return GameAnalysisResponse(
        games=games,
        analysis=result.analysis,
        model=result.model,
        tokens_used=result.tokens_used,
        game_count=len(games),
    )


async def analyze_player(
    player_name: str,
    season: int = _DEFAULT_SEASON,
) -> dict[str, Any]:
    """
    Generate a full player analysis for a given season.

    Retrieves game logs and official season averages, computes a recent-form
    window, and sends the assembled stat block to Claude for expert analysis.

    Parameters
    ----------
    player_name:
        Full or partial player name.
    season:
        NBA season start year.

    Returns
    -------
    dict
        Player metadata, season and recent-form averages, analysis text, and
        token usage. Returns ``{"error": "..."}`` on lookup failure.
    """
    logger.info("Analyzing player | name=%r season=%d", player_name, season)

    try:
        player, agg = await _build_player_stat_block(player_name, season)
    except ValueError as exc:
        logger.warning("Player lookup failed | name=%r error=%s", player_name, exc)
        return {"error": str(exc)}

    if agg["total_games"] == 0:
        return {
            "player": player.model_dump(),
            "analysis": (
                f"No stats found for {player.first_name} {player.last_name} "
                f"in the {season} season."
            ),
            "stats_count": 0,
        }

    stat_block = _render_stat_block(player, season, agg)

    result = await claude_service.analyze(
        prompt=f"Analyze this player's {season} NBA season:\n\n{stat_block}",
        system_prompt=NBA_ANALYST_SYSTEM_PROMPT,
    )

    logger.info(
        "Player analysis complete | player=%s %s tokens=%d",
        player.first_name,
        player.last_name,
        result.tokens_used,
    )

    return {
        "player": player.model_dump(),
        "season": season,
        "averages": {
            "points": agg["avg_pts"],
            "rebounds": agg["avg_reb"],
            "assists": agg["avg_ast"],
            "steals": agg["avg_stl"],
            "blocks": agg["avg_blk"],
            "fg_pct": agg["avg_fg"],
            "fg3_pct": agg["avg_fg3"],
            "ft_pct": agg["avg_ft"],
        },
        "last_10": {
            "points": agg["recent_pts"],
            "rebounds": agg["recent_reb"],
            "assists": agg["recent_ast"],
            "steals": agg["recent_stl"],
            "blocks": agg["recent_blk"],
            "fg_pct": agg["recent_fg"],
            "fg3_pct": agg["recent_fg3"],
        },
        "games_played": agg["total_games"],
        "analysis": result.analysis,
        "model": result.model,
        "tokens_used": result.tokens_used,
    }


async def analyze_player_section(
    player_name: str,
    season: int,
    section: str,
) -> dict[str, Any]:
    """
    Generate a focused single-section analysis (offense, defense, financials, etc.)
    for a given player.

    Parameters
    ----------
    player_name:
        Full or partial player name.
    season:
        NBA season start year.
    section:
        Analysis section key. Must be one of the keys in ``SECTION_PROMPTS``.

    Returns
    -------
    dict
        Player metadata, section identifier, analysis text, and token usage.
        Returns ``{"error": "..."}`` on lookup failure or unknown section.
    """
    logger.info(
        "Analyzing player section | name=%r season=%d section=%s",
        player_name,
        season,
        section,
    )

    if section not in SECTION_PROMPTS:
        valid_sections = ", ".join(sorted(SECTION_PROMPTS.keys()))
        return {
            "error": (
                f"Unknown section '{section}'. Valid sections: {valid_sections}"
            )
        }

    try:
        player, agg = await _build_player_stat_block(player_name, season)
    except ValueError as exc:
        logger.warning("Player lookup failed | name=%r error=%s", player_name, exc)
        return {"error": str(exc)}

    stat_block = _render_stat_block(player, season, agg)
    section_directive = SECTION_PROMPTS[section]

    prompt = (
        f"{section_directive}\n\n"
        f"Player data:\n{stat_block}\n\n"
        f"Go deep. This is a premium product. Use your basketball knowledge "
        f"beyond just the raw stats provided."
    )

    result = await claude_service.analyze(
        prompt=prompt,
        system_prompt=NBA_ANALYST_SYSTEM_PROMPT,
    )

    logger.info(
        "Player section analysis complete | player=%s %s section=%s tokens=%d",
        player.first_name,
        player.last_name,
        section,
        result.tokens_used,
    )

    return {
        "player": player.model_dump(),
        "section": section,
        "season": season,
        "analysis": result.analysis,
        "model": result.model,
        "tokens_used": result.tokens_used,
    }


async def analyze_trade(body: dict[str, Any]) -> dict[str, Any]:
    """
    Evaluate a proposed NBA trade from a front-office perspective.

    Attempts to fetch live stats for up to four named players involved in the
    trade. Stats are embedded in the prompt as supporting context. Pick assets
    (e.g. "2027 1st-round pick") are passed through as-is without a stats lookup.

    Parameters
    ----------
    body:
        Request body containing:
        - ``team_a`` (str): First team name.
        - ``team_b`` (str): Second team name.
        - ``team_a_gives`` (list[str]): Assets team A sends.
        - ``team_b_gives`` (list[str]): Assets team B sends.
        - ``context`` (str, optional): Additional context for the analysis.

    Returns
    -------
    dict
        Trade summary, per-player stats fetched, analysis text, and token usage.
    """
    team_a: str = body.get("team_a") or "Team A"
    team_b: str = body.get("team_b") or "Team B"
    team_a_gives: list[str] = body.get("team_a_gives") or []
    team_b_gives: list[str] = body.get("team_b_gives") or []
    context: str = body.get("context") or ""

    logger.info(
        "Analyzing trade | %s sends %s | %s sends %s",
        team_a,
        team_a_gives,
        team_b,
        team_b_gives,
    )

    # Fetch stats only for named players, skipping pick assets.
    all_assets: list[str] = team_a_gives + team_b_gives
    named_players = [p for p in all_assets if "pick" not in p.lower()][:_MAX_TRADE_PLAYERS]

    player_stats: dict[str, str] = {}

    for name in named_players:
        try:
            _, agg = await _build_player_stat_block(name, _DEFAULT_SEASON)
            if agg["total_games"] > 0:
                player_stats[name] = (
                    f"{agg['avg_pts']}pts / {agg['avg_reb']}reb / {agg['avg_ast']}ast "
                    f"({agg['total_games']}GP, {_DEFAULT_SEASON} season)"
                )
        except ValueError:
            logger.debug("No stats found for trade asset %r — skipping", name)
        except Exception as exc:
            logger.warning("Unexpected error fetching stats for %r: %s", name, exc)

    def _format_trade_side(team: str, gives: list[str]) -> str:
        lines = [f"{team} sends:"]
        for asset in gives:
            stat = player_stats.get(asset, "")
            stat_suffix = f"  [{stat}]" if stat else ""
            lines.append(f"  - {asset}{stat_suffix}")
        return "\n".join(lines)

    trade_block = (
        f"{_format_trade_side(team_a, team_a_gives)}\n\n"
        f"{_format_trade_side(team_b, team_b_gives)}"
    )

    if context:
        trade_block += f"\n\nAdditional context: {context}"

    prompt = (
        f"Evaluate this proposed NBA trade:\n\n"
        f"{trade_block}\n\n"
        f"Give a complete front office analysis. Who wins this trade and why? "
        f"What are the risks? What is your verdict?"
    )

    result = await claude_service.analyze(
        prompt=prompt,
        system_prompt=FRONT_OFFICE_SYSTEM_PROMPT,
    )

    logger.info("Trade analysis complete | tokens=%d", result.tokens_used)

    return {
        "team_a": team_a,
        "team_b": team_b,
        "team_a_gives": team_a_gives,
        "team_b_gives": team_b_gives,
        "player_stats": player_stats,
        "analysis": result.analysis,
        "model": result.model,
        "tokens_used": result.tokens_used,
    }


async def analyze_roster(team_name: str) -> dict[str, Any]:
    """
    Generate a front-office assessment of a team's roster, cap situation,
    and strategic priorities.

    Parameters
    ----------
    team_name:
        Full team name, city, or abbreviation. Used for both team lookup and
        as context in the prompt.

    Returns
    -------
    dict
        Team metadata (if found in the team list), analysis text, and token usage.
    """
    logger.info("Analyzing roster | team=%r", team_name)

    teams = await nba_service.get_all_teams()
    name_lower = team_name.lower()

    matched_team = next(
        (
            t for t in teams
            if name_lower in t.name.lower()
            or name_lower in t.city.lower()
            or name_lower in t.abbreviation.lower()
        ),
        None,
    )

    if matched_team:
        logger.debug("Matched team lookup | query=%r matched=%s", team_name, matched_team.name)
    else:
        logger.debug("No exact team match for %r; proceeding with prompt as-is", team_name)

    prompt = (
        f"Provide a comprehensive front office analysis for the {team_name}.\n\n"
        f"Cover:\n"
        f"1. Current roster assessment — who are the core pieces, who is expendable\n"
        f"2. Cap situation — are they over or under the cap, any bad contracts\n"
        f"3. Biggest roster need right now\n"
        f"4. Top 2 moves the front office should make this offseason\n"
        f"5. Trade candidates — who has value to other teams\n\n"
        f"Use your knowledge of the current {_DEFAULT_SEASON} NBA season."
    )

    result = await claude_service.analyze(
        prompt=prompt,
        system_prompt=FRONT_OFFICE_SYSTEM_PROMPT,
    )

    logger.info("Roster analysis complete | team=%r tokens=%d", team_name, result.tokens_used)

    return {
        "team": team_name,
        "team_data": matched_team.model_dump() if matched_team else None,
        "analysis": result.analysis,
        "model": result.model,
        "tokens_used": result.tokens_used,
    }


async def coach_adjustment(body: dict[str, Any]) -> dict[str, Any]:
    """
    Generate live in-game coaching adjustments based on the current box score.

    Fetches game context and player stat lines concurrently, then asks Claude
    to prescribe specific lineup, offensive, and defensive adjustments based
    on the actual live data.

    Parameters
    ----------
    body:
        Request body containing:
        - ``game_id`` (int, optional): BallDontLie game ID for live data.
        - ``situation`` (str, optional): Coach's free-text situation description.
        - ``my_team`` (str, optional): Team name for score-differential framing.

    Returns
    -------
    dict
        Game context used, box score availability flag, analysis text, and
        token usage.
    """
    game_id: Optional[int] = body.get("game_id")
    situation: str = body.get("situation") or ""
    my_team: str = body.get("my_team") or ""

    logger.info(
        "Coach adjustment | game_id=%s my_team=%r situation=%r",
        game_id,
        my_team,
        situation[:80] if situation else "",
    )

    game_context: str = ""
    box_summary: str = "Box score unavailable."
    box_available: bool = False

    if game_id:
        # -----------------------------------------------------------------------
        # Live game context — score, period, differential
        # -----------------------------------------------------------------------
        try:
            today = datetime.date.today().isoformat()
            games = await nba_service.get_games_by_date(today)
            game = next((g for g in games if g.id == game_id), None)

            if game:
                period = game.period or 0
                if period == 0:
                    quarter_label = "Pre-game"
                elif period <= 4:
                    quarter_label = f"Q{period}"
                else:
                    ot_num = period - 4
                    quarter_label = "OT" if ot_num == 1 else f"OT{ot_num}"

                home = game.home_team
                away = game.visitor_team
                home_score = game.home_team_score
                away_score = game.visitor_team_score

                is_home = (
                    my_team.lower() in home.name.lower()
                    or my_team.lower() in home.city.lower()
                    or my_team.lower() in home.abbreviation.lower()
                )

                my_score = home_score if is_home else away_score
                opp_score = away_score if is_home else home_score
                diff = my_score - opp_score

                if diff > 0:
                    diff_str = f"UP {abs(diff)}"
                elif diff < 0:
                    diff_str = f"DOWN {abs(diff)}"
                else:
                    diff_str = "TIED"

                game_context = (
                    f"GAME: {away.city} {away.name} ({away_score}) @ "
                    f"{home.city} {home.name} ({home_score})\n"
                    f"PERIOD: {quarter_label} | MY TEAM ({my_team}): {diff_str}\n"
                    f"STATUS: {game.status}"
                )
        except Exception as exc:
            logger.warning("Failed to fetch live game context | game_id=%s error=%s", game_id, exc)

        # -----------------------------------------------------------------------
        # Box score
        # -----------------------------------------------------------------------
        try:
            box = await nba_service.get_game_boxscore(game_id)

            if box and box.get("total_players", 0) > 0:
                box_available = True

                def _fmt_player_lines(players: list[dict], label: str) -> str:
                    header = f"\n{label}:"
                    lines = [header]
                    for p in players[:10]:
                        lines.append(
                            f"  {p['player']} ({p['pos']}): "
                            f"{p['pts']}pts {p['reb']}reb {p['ast']}ast "
                            f"{p['stl']}stl {p['blk']}blk "
                            f"{p['fg']}FG {p['fg3']}3P "
                            f"{p['min']}min {p['to']}TO {p['pf']}PF"
                        )
                    return "\n".join(lines)

                home_t = box.get("home_team") or {}
                away_t = box.get("away_team") or {}
                home_label = (home_t.get("name") or "Home") + " (HOME)"
                away_label = (away_t.get("name") or "Away") + " (AWAY)"

                box_summary = (
                    "FULL BOX SCORE:"
                    + _fmt_player_lines(box.get("home_players") or [], home_label)
                    + _fmt_player_lines(box.get("away_players") or [], away_label)
                )
        except Exception as exc:
            logger.warning("Failed to fetch box score | game_id=%s error=%s", game_id, exc)

    situation_line = (
        situation
        or "Give me the most important adjustments based on what you see in the box score right now."
    )

    prompt = (
        f"LIVE IN-GAME COACHING CALL — You have full situational awareness. "
        f"Do NOT ask for more information. Give adjustments immediately based on what you see.\n\n"
        f"{game_context}\n\n"
        f"COACH'S NOTE: {situation_line}\n\n"
        f"{box_summary}\n\n"
        f"Based on EVERYTHING above — the live score, who's hot, who's struggling, shooting splits, "
        f"foul trouble, minutes load, turnovers — give me:\n"
        f"1. The single most important adjustment RIGHT NOW (name players, name schemes)\n"
        f"2. Lineup change if needed (who in, who out, why)\n"
        f"3. Defensive adjustment based on what their guys are doing\n"
        f"4. One offensive action to run next possession\n\n"
        f"You have the data. Use it. Be surgical."
    )

    result = await claude_service.analyze(
        prompt=prompt,
        system_prompt=COACH_SYSTEM_PROMPT,
    )

    logger.info(
        "Coach adjustment complete | game_id=%s box_used=%s tokens=%d",
        game_id,
        box_available,
        result.tokens_used,
    )

    return {
        "game_id": game_id,
        "my_team": my_team,
        "situation": situation,
        "box_score_used": box_available,
        "analysis": result.analysis,
        "model": result.model,
        "tokens_used": result.tokens_used,
    }


async def timeout_play(body: dict[str, Any]) -> dict[str, Any]:
    """
    Draw up a specific in-bounds or half-court play for use coming out of a timeout.

    Optionally fetches the live box score to inform which players are available
    and hot vs. cold, then asks Claude to design an executable play with primary
    and secondary options.

    Parameters
    ----------
    body:
        Request body containing:
        - ``game_id`` (int, optional): BallDontLie game ID for live data.
        - ``my_team`` (str, optional): Team name for roster filtering.
        - ``score_diff`` (int, optional): Score differential (positive = leading).
        - ``time_remaining`` (str, optional): Time remaining, e.g. ``"0:24"``.
        - ``quarter`` (int, optional): Current quarter / period number.
        - ``situation`` (str, optional): Additional situational context.

    Returns
    -------
    dict
        Situation context, drawn-up play text, model metadata, and token usage.
    """
    game_id: Optional[int] = body.get("game_id")
    my_team: str = body.get("my_team") or ""
    score_diff: int = int(body.get("score_diff") or 0)
    time_remaining: str = body.get("time_remaining") or ""
    quarter: int = int(body.get("quarter") or 4)
    situation: str = body.get("situation") or ""

    logger.info(
        "Timeout play | game_id=%s team=%r Q%d %s diff=%+d",
        game_id,
        my_team,
        quarter,
        time_remaining,
        score_diff,
    )

    box_summary: str = ""

    if game_id:
        try:
            box = await nba_service.get_game_boxscore(game_id)

            if box and box.get("total_players", 0) > 0:
                home_t = box.get("home_team") or {}
                home_name_lower = (home_t.get("name") or "").lower()
                is_home = my_team.lower() in home_name_lower

                my_players_key = "home_players" if is_home else "away_players"
                my_players: list[dict] = (box.get(my_players_key) or [])[:8]

                # Sort by minutes played descending so we see the rotation players first.
                my_players_sorted = sorted(
                    my_players,
                    key=lambda p: str(p.get("min") or "0"),
                    reverse=True,
                )

                player_lines = "\n".join(
                    f"  {p['player']} ({p['pos']}): "
                    f"{p['pts']}pts {p['reb']}reb {p['ast']}ast "
                    f"{p['fg']}FG {p['fg3']}3P {p['min']}min {p['pf']}PF"
                    for p in my_players_sorted
                )

                box_summary = f"My active players:\n{player_lines}"
        except Exception as exc:
            logger.warning(
                "Failed to fetch box score for timeout play | game_id=%s error=%s",
                game_id,
                exc,
            )

    if score_diff > 0:
        diff_str = f"up {abs(score_diff)}"
    elif score_diff < 0:
        diff_str = f"down {abs(score_diff)}"
    else:
        diff_str = "tied"

    prompt_parts = [
        f"TIMEOUT — Draw up a play. I need something executable in 20 seconds.\n",
        f"Team: {my_team or 'My team'}",
        f"Situation: Q{quarter}, {time_remaining} remaining, {diff_str}",
    ]
    if situation:
        prompt_parts.append(f"Context: {situation}")
    if box_summary:
        prompt_parts.append(box_summary)

    prompt_parts += [
        "",
        "Give me:",
        "1. Play name",
        "2. Setup and motion",
        "3. Primary option",
        "4. Secondary option if primary is denied",
        "5. One sentence on why this works against what they are likely running defensively",
    ]

    prompt = "\n".join(prompt_parts)

    result = await claude_service.analyze(
        prompt=prompt,
        system_prompt=COACH_SYSTEM_PROMPT,
    )

    logger.info(
        "Timeout play complete | game_id=%s team=%r tokens=%d",
        game_id,
        my_team,
        result.tokens_used,
    )

    return {
        "game_id": game_id,
        "my_team": my_team,
        "quarter": quarter,
        "time_remaining": time_remaining,
        "score_diff": score_diff,
        "play": result.analysis,
        "model": result.model,
        "tokens_used": result.tokens_used,
    }
