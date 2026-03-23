from fastapi import APIRouter, HTTPException, Query, Body, Response
from typing import Optional, List

from app.services import nba_service, claude_service, analysis_service
from app.models.schemas import (
    AnalysisRequest,
    AnalysisResponse,
    GameAnalysisResponse,
)
from app.utils.helpers import validate_date_string

router = APIRouter()


# ── Health ────────────────────────────────────────────────────────────────────

@router.get("/health")
async def health():
    from app.core.config import get_settings
    settings = get_settings()
    return {"status": "ok", "environment": settings.environment, "version": "1.0.0"}


# ── NBA ───────────────────────────────────────────────────────────────────────

@router.get("/nba/games")
async def get_games(response: Response, date: Optional[str] = Query(None, description="Format: YYYY-MM-DD")):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
    if date and not validate_date_string(date):
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")
    try:
        games = await nba_service.get_games_by_date(date)
        return {"games": [g.model_dump() for g in games], "count": len(games)}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"BallDontLie API error: {str(e)}")


@router.get("/nba/teams")
async def get_teams():
    try:
        teams = await nba_service.get_all_teams()
        return {"teams": [t.model_dump() for t in teams], "count": len(teams)}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"BallDontLie API error: {str(e)}")


@router.get("/nba/teams/{team_id}")
async def get_team(team_id: int):
    try:
        team = await nba_service.get_team_by_id(team_id)
        return team.model_dump()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"BallDontLie API error: {str(e)}")


@router.get("/nba/players")
async def search_players(name: str = Query(..., description="Player name to search")):
    try:
        players = await nba_service.search_players(name)
        return {"players": [p.model_dump() for p in players], "count": len(players)}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"BallDontLie API error: {str(e)}")


@router.get("/nba/games/{game_id}/boxscore")
async def get_game_boxscore(game_id: int):
    try:
        return await nba_service.get_game_boxscore(game_id)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"BallDontLie API error: {str(e)}")


@router.get("/nba/players/{player_id}/stats")
async def get_player_stats(player_id: int, season: int = Query(2025, description="NBA season year")):
    try:
        stats = await nba_service.get_player_stats(player_id, season)
        return {"stats": [s.model_dump() for s in stats], "count": len(stats)}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"BallDontLie API error: {str(e)}")


# ── Claude ────────────────────────────────────────────────────────────────────

@router.post("/claude/analyze", response_model=AnalysisResponse)
async def claude_analyze(body: AnalysisRequest):
    try:
        result = await claude_service.analyze(
            prompt=body.prompt,
            system_prompt=body.context or "",
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Claude API error: {str(e)}")


# ── Analysis (Combined) ───────────────────────────────────────────────────────

@router.get("/analysis/today-games", response_model=GameAnalysisResponse)
async def today_games_analysis(date: Optional[str] = Query(None, description="Format: YYYY-MM-DD")):
    if date and not validate_date_string(date):
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")
    try:
        return await analysis_service.analyze_today_games(date)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


@router.get("/analysis/player")
async def player_analysis(
    name: str = Query(..., description="Player name"),
    season: int = Query(2025, description="NBA season year"),
):
    try:
        return await analysis_service.analyze_player(name, season)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


@router.get("/analysis/player/section")
async def player_section_analysis(
    name: str = Query(..., description="Player name"),
    season: int = Query(2025, description="NBA season year"),
    section: str = Query(..., description="Section: offense|defense|off_the_court|injuries|financials"),
):
    try:
        return await analysis_service.analyze_player_section(name, season, section)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


# ── Front Office ──────────────────────────────────────────────────────────────

@router.post("/frontoffice/trade")
async def analyze_trade(body: dict = Body(...)):
    """
    Analyze a proposed trade. Expects:
    {
      "team_a": "Lakers",
      "team_b": "Celtics",
      "team_a_gives": ["Player Name", "2025 First Round Pick"],
      "team_b_gives": ["Player Name", "Player Name"],
      "context": "optional extra context"
    }
    """
    try:
        return await analysis_service.analyze_trade(body)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


@router.get("/frontoffice/roster")
async def get_roster_analysis(team_name: str = Query(..., description="Team name")):
    """Get roster breakdown and financial analysis for a team."""
    try:
        return await analysis_service.analyze_roster(team_name)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


# ── Coach Mode ────────────────────────────────────────────────────────────────

@router.post("/coach/adjust")
async def coach_adjustment(body: dict = Body(...)):
    """
    Real-time coaching adjustment. Expects:
    {
      "game_id": 12345,
      "situation": "We're down 8 in the 3rd, their zone is killing us",
      "my_team": "Lakers"
    }
    """
    try:
        return await analysis_service.coach_adjustment(body)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


@router.post("/coach/timeout")
async def timeout_play(body: dict = Body(...)):
    """
    Generate a timeout play. Expects:
    {
      "game_id": 12345,
      "my_team": "Lakers",
      "score_diff": -4,
      "time_remaining": "2:30",
      "quarter": 4,
      "situation": "Need a quick bucket, they're playing man"
    }
    """
    try:
        return await analysis_service.timeout_play(body)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

