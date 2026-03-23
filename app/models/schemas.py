from pydantic import BaseModel, Field
from typing import Optional, List


# ---------------------------------------------------------------------------
# Core Domain Models
# ---------------------------------------------------------------------------

class Team(BaseModel):
    """NBA team metadata."""
    id: int
    name: str
    abbreviation: str
    city: str
    conference: str
    division: str


class Game(BaseModel):
    """Represents a single NBA game."""
    id: int
    date: str
    status: str

    period: Optional[int] = None
    time: Optional[str] = None

    home_team: Team
    visitor_team: Team

    home_team_score: int
    visitor_team_score: int


class Player(BaseModel):
    """Basic player profile."""
    id: int
    first_name: str
    last_name: str

    position: Optional[str] = None
    team: Optional[Team] = None


class PlayerStats(BaseModel):
    """Per-game statistical line for a player."""
    player: Player
    game_id: int

    points: int
    rebounds: int
    assists: int
    steals: int
    blocks: int

    minutes: Optional[str] = None

    fg_pct: Optional[float] = Field(
        default=None,
        description="Field goal percentage"
    )
    fg3_pct: Optional[float] = Field(
        default=None,
        description="Three-point field goal percentage"
    )
    ft_pct: Optional[float] = Field(
        default=None,
        description="Free throw percentage"
    )


# ---------------------------------------------------------------------------
# Analysis API Models
# ---------------------------------------------------------------------------

class AnalysisRequest(BaseModel):
    """Request payload for AI analysis endpoints."""
    prompt: str
    context: Optional[str] = None


class AnalysisResponse(BaseModel):
    """Standard response from analysis engine."""
    analysis: str
    model: str
    tokens_used: int


class GameAnalysisResponse(BaseModel):
    """Analysis response containing game data."""
    games: List[Game]

    analysis: str
    model: str
    tokens_used: int

    game_count: int


# ---------------------------------------------------------------------------
# System / Operational Models
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    """Service health check response."""
    status: str
    environment: str
    version: str = "1.0.0"