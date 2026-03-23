from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router
from app.core.config import get_settings
from app.core.http_client import GlobalHTTPClient

settings = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize the global HTTP client pool
    await GlobalHTTPClient.start()
    
    yield  # The application runs while yielded
    
    # Shutdown: Close connections gracefully
    await GlobalHTTPClient.stop()

app = FastAPI(
    title="PIVOT",
    description="NBA intelligence platform powered by Claude + BallDontLie",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,  # <-- Added the lifespan hook here
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
    return {
        "app": "PIVOT",
        "status": "running",
        "docs": "/docs",
        "environment": settings.environment,
    }