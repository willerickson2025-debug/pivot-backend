import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from app.api.routes import router
from app.core.config import get_settings

settings = get_settings()

app = FastAPI(
    title="PIVOT",
    description="NBA intelligence platform powered by Claude + BallDontLie",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")

@app.get("/")
async def root():
    dashboard = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dashboard.html")
    if os.path.exists(dashboard):
        return FileResponse(dashboard)
    return {"app": "PIVOT", "status": "running", "docs": "/docs", "environment": settings.environment}

@app.get("/health")
async def health():
    return {"status": "ok", "environment": settings.environment, "version": "1.0.0"}
