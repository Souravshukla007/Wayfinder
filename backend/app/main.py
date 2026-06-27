"""Wayfinder FastAPI application entrypoint.

Creates the app, configures CORS, registers routers (added incrementally as the
backend is built out), and exposes a lifespan + health check.

Requirements: 20.2 (FastAPI + Pydantic v2).
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: provider registry, db/redis pools, JWKS warm-up are wired here
    # as their tasks land. Kept minimal for the scaffold.
    yield
    # Shutdown: release pools/connections.


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health", tags=["system"])
    def health() -> dict[str, str]:
        return {"status": "ok", "app": settings.app_name, "env": settings.environment}

    # Routers (registered as their tasks are implemented):
    from app.api import routes_preferences, routes_trips, routes_ws

    app.include_router(routes_trips.router)
    app.include_router(routes_preferences.router)
    app.include_router(routes_ws.router)

    return app


app = create_app()
