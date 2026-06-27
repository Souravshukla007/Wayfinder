"""Vercel FastAPI entrypoint.

Vercel's FastAPI runtime discovers the ASGI ``app`` instance in a file named
``app.py``/``index.py``/``server.py``/``main.py``/``wsgi.py``/``asgi.py`` at the
service root (here: ``backend/``). The actual application is defined in
``app/main.py``; this module simply re-exports it so Vercel can find it.
"""

from __future__ import annotations

from app.main import app

__all__ = ["app"]
