"""SQLAlchemy declarative base for all Wayfinder ORM models.

A single :class:`Base` is defined here and shared by every model in
:mod:`app.models.db`. Keeping the base in its own module avoids circular
imports (e.g. between ORM models and Alembic migration metadata) and gives
session/engine wiring (task 2.3) a single ``Base.metadata`` to target.
"""

from __future__ import annotations

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Declarative base shared by all ORM models.

    All tables defined against this base are reachable via ``Base.metadata``,
    which the engine/migration layer (task 2.3) uses to create or migrate the
    schema. SQLAlchemy 2.x typed declarative mapping (``Mapped`` /
    ``mapped_column``) is used throughout the models module.
    """
