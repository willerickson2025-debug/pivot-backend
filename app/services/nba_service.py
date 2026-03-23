"""
nba_service.py
==============
Data access layer for the BallDontLie NBA API.

Responsibilities
----------------
- All HTTP communication with the BallDontLie REST API
- Domain object hydration (raw dict → typed schema)
- Retry / timeout / error-propagation policy
- Box score aggregation and player-stat retrieval

This module is intentionally free of business logic. Analysis logic lives in
analysis_service.py; Claude integration lives in claude_service.py.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any, Optional
from zoneinfo import ZoneInfo

import httpx

from app.core.config import get_settings
from app.models.schemas import Game, Player, PlayerStats, Team

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_PER_PAGE: int = 100
_REQUEST_TIMEOUT: float = 12.0          # seconds per attempt
_MAX_RETRIES: int = 3                   # total attempts before raising
_RETRY_BACKOFF_BASE: float = 0.5        # seconds; multiplied by attempt index
_DEFAULT_SEASON: int = 2025
_CENTRAL_TZ: str = "America/Chicago"


# ---------------------------------------------------------------------------
# Internal HTTP Layer
# ---------------------------------------------------------------------------

async def _fetch_data(
    endpoint: str,
    params: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Execute an authenticated GET request against the BallDontLie API.

    Centralises:
    - Base URL & authentication header injection
    - Per-request timeout enforcement
    - Exponential-ish back-off retry on transient network/server errors
    - Structured logging of every outbound request and its outcome
    - JSON decoding with a meaningful error on malformed payloads

    Parameters
    ----------
    endpoint:
        Path relative to the configured base URL, e.g. ``"/games"``.
    params:
        Optional query-string parameters forwarded verbatim to httpx.

    Returns
    -------
    dict
        Parsed JSON payload from the API response body.

    Raises
    ------
    httpx.HTTPStatusError
        Propagated after all retries are exhausted for 4xx/5xx responses.
    httpx.RequestError
        Propagated after all retries are exhausted for connection failures.
    ValueError
        If the response body cannot be decoded as JSON.
    """
    settings = get_settings()
    clean_params = params or {}

    last_exc: Exception | None = None

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            logger.debug(
                "BallDontLie request | attempt=%d endpoint=%s params=%s",
                attempt,
                endpoint,
                clean_params,
            )

            async with httpx.AsyncClient(
                base_url=settings.balldontlie_base_url,
                timeout=_REQUEST_TIMEOUT,
            ) as client:
                response = await client.get(
                    endpoint,
                    headers={"Authorization": settings.balldontlie_api_key},
                    params=clean_params,
                )

            response.raise_for_status()

            try:
                payload: dict[str, Any] = response.json()
            except Exception as exc:
                raise ValueError(
                    f"BallDontLie returned non-JSON body for {endpoint}: "
                    f"{response.text[:200]}"
                ) from exc

            logger.debug(
                "BallDontLie response | endpoint=%s status=%d",
                endpoint,
                response.status_code,
            )
            return payload

        except (httpx.TransportError, httpx.TimeoutException) as exc:
            last_exc = exc
            if attempt < _MAX_RETRIES:
                backoff = _RETRY_BACKOFF_BASE * attempt
                logger.warning(
                    "BallDontLie transient error | attempt=%d endpoint=%s error=%s | "
                    "retrying in %.1fs",
                    attempt,
                    endpoint,
                    exc,
                    backoff,
                )
                await asyncio.sleep(backoff)
            else:
                logger.error(
                    "BallDontLie request failed after %d attempts | endpoint=%s error=%s",
                    _MAX_RETRIES,
                    endpoint,
                    exc,
                )

        except httpx.HTTPStatusError as exc:
            # 4xx errors are not retried — retrying a 404 or 422 won't help.
            # 5xx errors could be retried, but we surface them immediately so
            # callers can make that decision at a higher layer.
            logger.error(
                "BallDontLie HTTP error | endpoint=%s status=%d body=%s",
                endpoint,
                exc.response.status_code,
                exc.response.text[:500],
            )
            raise

    # All retries exhausted for transient errors
    assert last_exc is not None
    raise last_exc


# ---------------------------------------------------------------------------
# Parsing Utilities
# ---------------------------------------------------------------------------

def _require(mapping: dict[str, Any], key: str, context: str = "") -> Any:
    """
    Return ``mapping[key]``, raising ``KeyError`` with a descriptive message
    if the key is absent. Used to surface schema mismatches early rather than
    producing silent ``None`` values downstream.
    """
    if key not in mapping:
        location = f" (in {context})" if context else ""
        raise KeyError(
            f"Expected key '{key}' missing from BallDontLie payload{location}. "
            f"Available keys: {list(mapping.keys())}"
        )
    return mapping[key]


def _parse_team(raw: dict[str, Any]) -> Team:
    """
    Hydrate a ``Team`` domain object from a raw BallDontLie team payload.

    The API occasionally returns partial team objects inside game payloads
    (e.g. missing ``full_name``). We prefer ``full_name`` but fall back to
    composing it from ``city`` + ``name`` if needed.
    """
    team_id: int = _require(raw, "id", "team")
    city: str = raw.get("city") or ""
    name: str = raw.get("name") or ""

    return Team(
        id=team_id,
        name=raw.get("full_name") or f"{city} {name}".strip() or name,
        abbreviation=raw.get("abbreviation") or "",
        city=city,
        conference=raw.get("conference") or "",
        division=raw.get("division") or "",
    )


def _parse_game(raw: dict[str, Any]) -> Game:
    """
    Hydrate a ``Game`` domain object from a raw BallDontLie game payload.

    Scores default to 0 rather than ``None`` so callers can do arithmetic
    without null-checks everywhere.
    """
    return Game(
        id=_require(raw, "id", "game"),
        date=raw.get("date") or "",
        status=raw.get("status") or "Unknown",
        period=raw.get("period"),
        time=raw.get("time"),
        home_team=_parse_team(_require(raw, "home_team", "game")),
        visitor_team=_parse_team(_require(raw, "visitor_team", "game")),
        home_team_score=int(raw.get("home_team_score") or 0),
        visitor_team_score=int(raw.get("visitor_team_score") or 0),
    )


def _parse_player(raw: dict[str, Any]) -> Player:
    """
    Hydrate a ``Player`` domain object from a raw BallDontLie player payload.

    The nested ``team`` object is optional — players without a current team
    assignment (free agents, two-way contracts in flux) are handled gracefully.
    """
    team_raw: dict[str, Any] | None = raw.get("team")

    return Player(
        id=_require(raw, "id", "player"),
        first_name=raw.get("first_name") or "",
        last_name=raw.get("last_name") or "",
        position=raw.get("position") or None,
        team=_parse_team(team_raw) if team_raw else None,
    )


def _parse_stat_line(s: dict[str, Any]) -> dict[str, Any]:
    """
    Convert a single raw stat entry from the BallDontLie ``/stats`` endpoint
    into a normalised, display-ready dict.

    All numeric fields default to 0 and all string fields to sensible
    placeholders so renderers never encounter ``None``.
    """
    player_raw: dict[str, Any] = s.get("player") or {}
    team_raw: dict[str, Any] = s.get("team") or {}

    first = player_raw.get("first_name") or ""
    last = player_raw.get("last_name") or ""
    full_name = f"{first} {last}".strip() or "Unknown"

    fgm = int(s.get("fgm") or 0)
    fga = int(s.get("fga") or 0)
    fg3m = int(s.get("fg3m") or 0)
    fg3a = int(s.get("fg3a") or 0)
    ftm = int(s.get("ftm") or 0)
    fta = int(s.get("fta") or 0)

    return {
        "player": full_name,
        "pos": player_raw.get("position") or "—",
        "min": s.get("min") or "0",
        "pts": int(s.get("pts") or 0),
        "reb": int(s.get("reb") or 0),
        "ast": int(s.get("ast") or 0),
        "stl": int(s.get("stl") or 0),
        "blk": int(s.get("blk") or 0),
        "fg": f"{fgm}-{fga}",
        "fg3": f"{fg3m}-{fg3a}",
        "ft": f"{ftm}-{fta}",
        "fg_pct": round(fgm / fga, 3) if fga > 0 else 0.0,
        "fg3_pct": round(fg3m / fg3a, 3) if fg3a > 0 else 0.0,
        "ft_pct": round(ftm / fta, 3) if fta > 0 else 0.0,
        "to": int(s.get("turnover") or 0),
        "pf": int(s.get("pf") or 0),
        "plus_minus": int(s.get("plus_minus") or 0),
        "team_id": team_raw.get("id"),
        "team_abbr": team_raw.get("abbreviation") or "",
    }


# ---------------------------------------------------------------------------
# Public Service Layer — Queries
# ---------------------------------------------------------------------------

async def get_games_by_date(target_date: Optional[str] = None) -> list[Game]:
    """
    Fetch all NBA games scheduled for a specific calendar date.

    Parameters
    ----------
    target_date:
        ISO-8601 date string (``"YYYY-MM-DD"``). Defaults to the current date
        in US Central Time when omitted, which aligns with the NBA schedule
        timezone used by BallDontLie.

    Returns
    -------
    list[Game]
        All games found for the given date, or an empty list when there are
        none scheduled.
    """
    query_date = target_date or datetime.now(ZoneInfo(_CENTRAL_TZ)).strftime("%Y-%m-%d")

    logger.info("Fetching games for date=%s", query_date)

    payload = await _fetch_data(
        "/games",
        params={"dates[]": query_date, "per_page": _DEFAULT_PER_PAGE},
    )

    games = [_parse_game(g) for g in payload.get("data") or []]
    logger.info("Found %d game(s) for %s", len(games), query_date)
    return games


async def get_team_by_id(team_id: int) -> Team:
    """
    Retrieve a single NBA team by its BallDontLie team ID.

    Parameters
    ----------
    team_id:
        BallDontLie internal team identifier.

    Returns
    -------
    Team
        Hydrated Team domain object.
    """
    logger.debug("Fetching team id=%d", team_id)
    payload = await _fetch_data(f"/teams/{team_id}")
    return _parse_team(payload.get("data") or {})


async def get_all_teams() -> list[Team]:
    """
    Retrieve every NBA team from BallDontLie.

    Returns
    -------
    list[Team]
        All 30 franchises (and any G-League entries if present).
    """
    logger.debug("Fetching all teams")
    payload = await _fetch_data("/teams", params={"per_page": _DEFAULT_PER_PAGE})
    teams = [_parse_team(t) for t in payload.get("data") or []]
    logger.debug("Received %d teams", len(teams))
    return teams


async def search_players(name: str) -> list[Player]:
    """
    Search NBA players by full or partial name.

    The BallDontLie API performs server-side fuzzy matching on the ``search``
    parameter, so both ``"LeBron"`` and ``"lebron james"`` are valid inputs.

    Parameters
    ----------
    name:
        Full or partial player name. Case-insensitive.

    Returns
    -------
    list[Player]
        Matching players, ordered by API relevance. Empty list on no match.
    """
    logger.debug("Searching players | query=%r", name)
    payload = await _fetch_data(
        "/players",
        params={"search": name.strip(), "per_page": 25},
    )
    players = [_parse_player(p) for p in payload.get("data") or []]
    logger.debug("Player search for %r returned %d result(s)", name, len(players))
    return players


# ---------------------------------------------------------------------------
# Box Score Aggregation
# ---------------------------------------------------------------------------

async def get_game_boxscore(game_id: int) -> dict[str, Any]:
    """
    Retrieve and aggregate a full box score for a completed or in-progress game.

    Fires two concurrent API requests — one for game metadata and one for
    per-player statistics — and merges them into a single normalised payload.

    The returned dict is guaranteed to have all top-level keys present even
    when no player stats are available yet (e.g. a game that has not tipped off).

    Parameters
    ----------
    game_id:
        BallDontLie internal game identifier.

    Returns
    -------
    dict
        Keys: ``game_id``, ``game_info``, ``home_team``, ``away_team``,
        ``home_players``, ``away_players``, ``total_players``.
    """
    logger.info("Fetching box score for game_id=%d", game_id)

    # Fire both API calls concurrently to halve wall-clock latency.
    game_payload, stats_payload = await asyncio.gather(
        _fetch_data(f"/games/{game_id}"),
        _fetch_data("/stats", params={"game_ids[]": game_id, "per_page": _DEFAULT_PER_PAGE}),
    )

    game_raw: dict[str, Any] = game_payload.get("data") or {}

    # -----------------------------------------------------------------------
    # Game metadata
    # -----------------------------------------------------------------------

    game_info: dict[str, Any] = {
        "id": game_raw.get("id"),
        "date": game_raw.get("date") or "",
        "status": game_raw.get("status") or "Unknown",
        "period": game_raw.get("period"),
        "time": game_raw.get("time"),
        "home_team_score": int(game_raw.get("home_team_score") or 0),
        "away_team_score": int(game_raw.get("visitor_team_score") or 0),
    }

    # -----------------------------------------------------------------------
    # Team metadata
    # -----------------------------------------------------------------------

    home_raw: dict[str, Any] = game_raw.get("home_team") or {}
    away_raw: dict[str, Any] = game_raw.get("visitor_team") or {}

    home_city = home_raw.get("city") or ""
    home_name = home_raw.get("name") or ""
    away_city = away_raw.get("city") or ""
    away_name = away_raw.get("name") or ""

    home_team: dict[str, Any] = {
        "id": home_raw.get("id"),
        "name": home_raw.get("full_name") or f"{home_city} {home_name}".strip(),
        "abbreviation": home_raw.get("abbreviation") or "",
        "score": int(game_raw.get("home_team_score") or 0),
    }

    away_team: dict[str, Any] = {
        "id": away_raw.get("id"),
        "name": away_raw.get("full_name") or f"{away_city} {away_name}".strip(),
        "abbreviation": away_raw.get("abbreviation") or "",
        "score": int(game_raw.get("visitor_team_score") or 0),
    }

    # -----------------------------------------------------------------------
    # Player stat lines
    # -----------------------------------------------------------------------

    home_players: list[dict[str, Any]] = []
    away_players: list[dict[str, Any]] = []

    for s in stats_payload.get("data") or []:
        stat_line = _parse_stat_line(s)
        team_id = stat_line.get("team_id")

        if team_id == home_team["id"]:
            home_players.append(stat_line)
        elif team_id == away_team["id"]:
            away_players.append(stat_line)
        else:
            logger.debug(
                "Stat line for %r has unknown team_id=%s; skipping",
                stat_line["player"],
                team_id,
            )

    # Sort by points descending — highest scorers first in each list.
    home_players.sort(key=lambda x: x["pts"], reverse=True)
    away_players.sort(key=lambda x: x["pts"], reverse=True)

    total = len(home_players) + len(away_players)
    logger.info(
        "Box score assembled | game_id=%d players=%d home=%d away=%d",
        game_id,
        total,
        len(home_players),
        len(away_players),
    )

    return {
        "game_id": game_id,
        "game_info": game_info,
        "home_team": home_team,
        "away_team": away_team,
        "home_players": home_players,
        "away_players": away_players,
        "total_players": total,
    }


# ---------------------------------------------------------------------------
# Player Season Stats
# ---------------------------------------------------------------------------

async def get_player_stats(player_id: int, season: int = _DEFAULT_SEASON) -> list[PlayerStats]:
    """
    Retrieve per-game stat logs for a player for a given season.

    Results are sorted chronologically (ascending game ID) so callers can
    trivially slice ``[-10:]`` for a recent-form window.

    Parameters
    ----------
    player_id:
        BallDontLie internal player identifier.
    season:
        NBA season year (the year the season *starts* in; 2024 = 2024–25).

    Returns
    -------
    list[PlayerStats]
        One entry per game played. Empty list if no stats are found.
    """
    logger.debug("Fetching player stats | player_id=%d season=%d", player_id, season)

    payload = await _fetch_data(
        "/stats",
        params={
            "player_ids[]": player_id,
            "seasons[]": season,
            "per_page": _DEFAULT_PER_PAGE,
        },
    )

    results: list[PlayerStats] = []

    for s in payload.get("data") or []:
        player_raw: dict[str, Any] = s.get("player") or {}
        game_raw: dict[str, Any] = s.get("game") or {}

        results.append(
            PlayerStats(
                player=_parse_player(player_raw),
                game_id=int(game_raw.get("id") or 0),
                points=int(s.get("pts") or 0),
                rebounds=int(s.get("reb") or 0),
                assists=int(s.get("ast") or 0),
                steals=int(s.get("stl") or 0),
                blocks=int(s.get("blk") or 0),
                minutes=s.get("min"),
                fg_pct=s.get("fg_pct"),
                fg3_pct=s.get("fg3_pct"),
                ft_pct=s.get("ft_pct"),
            )
        )

    # Sort ascending by game_id so recent-form slicing (stats[-10:]) is valid.
    results.sort(key=lambda x: x.game_id)

    logger.debug(
        "Fetched %d stat entries | player_id=%d season=%d",
        len(results),
        player_id,
        season,
    )
    return results


async def get_season_averages(player_id: int, season: int = _DEFAULT_SEASON) -> dict[str, Any]:
    """
    Fetch official season averages for a player from BallDontLie.

    Uses the ``/season_averages`` endpoint which returns pre-computed averages
    that are more accurate than computing them from raw game logs (which can
    lag or omit double-headers).

    Parameters
    ----------
    player_id:
        BallDontLie internal player identifier.
    season:
        NBA season start year.

    Returns
    -------
    dict
        Season average dict from the API, or an empty dict if no data exists
        (e.g. the player did not play that season or the endpoint is unavailable).
    """
    logger.debug(
        "Fetching season averages | player_id=%d season=%d", player_id, season
    )

    try:
        payload = await _fetch_data(
            "/season_averages",
            params={"player_id": player_id, "season": season},
        )
        data: list[dict[str, Any]] = payload.get("data") or []

        if not data:
            logger.debug(
                "No season averages found | player_id=%d season=%d", player_id, season
            )
            return {}

        averages = data[0]
        logger.debug(
            "Season averages retrieved | player_id=%d pts=%.1f reb=%.1f ast=%.1f",
            player_id,
            averages.get("pts") or 0,
            averages.get("reb") or 0,
            averages.get("ast") or 0,
        )
        return averages

    except Exception as exc:
        # Season averages are supplementary data. A failure here degrades
        # gracefully — callers fall back to computing averages from game logs.
        logger.warning(
            "Season averages fetch failed | player_id=%d season=%d error=%s",
            player_id,
            season,
            exc,
        )
        return {}
