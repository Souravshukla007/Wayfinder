"""Database engine, session factory, and table creation helper.

The engine and session factory are created **lazily** (on first use) rather than
at import time. Importing this module — e.g. to depend on :func:`get_session`
from an API route — therefore does not eagerly connect a driver or require the
configured database's DBAPI to be installed. This keeps route modules importable
in environments (such as the unit-test suite) that override the session with an
in-memory database and never touch the configured engine.

``engine`` and ``SessionLocal`` remain accessible as module attributes for
backward compatibility; they are materialized on first access.

Requirements: 17.1, 17.2, 17.3.
"""

from __future__ import annotations

from collections.abc import Iterator
from functools import lru_cache

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings
from app.models.db import Base


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Return the process-wide SQLAlchemy engine (created on first call)."""
    settings = get_settings()
    url = settings.database_url
    if url.startswith("sqlite"):
        # Local-dev SQLite: allow connections to cross threads (FastAPI runs the
        # background planning job in a worker thread) and keep one shared
        # connection so an in-memory/file DB sees a consistent schema. Only the
        # sqlite path is affected; Postgres (prod) is unchanged.
        from sqlalchemy.pool import StaticPool

        return create_engine(
            url,
            future=True,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
    return create_engine(url, pool_pre_ping=True, future=True)


@lru_cache(maxsize=1)
def get_session_factory() -> sessionmaker[Session]:
    """Return the process-wide session factory bound to the engine."""
    return sessionmaker(
        bind=get_engine(), autoflush=False, expire_on_commit=False, class_=Session
    )


def create_all() -> None:
    """Create all tables. Used for local/dev bootstrap; Alembic owns prod migrations."""
    Base.metadata.create_all(bind=get_engine())


def get_session() -> Iterator[Session]:
    """FastAPI dependency that yields a scoped session and ensures cleanup."""
    session = get_session_factory()()
    try:
        yield session
    finally:
        session.close()


def __getattr__(name: str):
    """Lazily materialize ``engine`` / ``SessionLocal`` on attribute access.

    Preserves the previous module-level names without constructing the engine at
    import time (PEP 562 module ``__getattr__``).
    """
    if name == "engine":
        return get_engine()
    if name == "SessionLocal":
        return get_session_factory()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
