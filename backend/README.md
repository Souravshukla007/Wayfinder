# Wayfinder Travel Planner — Backend

FastAPI + Pydantic v2 backend for the Wayfinder multi-agent travel consultant.
It runs end-to-end on mock data with **zero paid API keys** and swaps in real
providers purely through environment configuration.

> This directory is a greenfield Python project. It is independent of the Vite
> React prototype at the repository root (`src/`).

## Layout

```
backend/
  app/
    main.py            FastAPI app, lifespan, router registration
    config.py          Pydantic Settings (env-driven provider/LLM selection)
    auth/              Supabase JWT verification (JWKS + require_user)
    api/               REST + WebSocket routers
    orchestration/     LangGraph graph, shared state, agent nodes
    decision/          Deterministic Travel Decision Engine + ledger
    solver/            CP-SAT constraint solver (OR-Tools)
    memory/            Travel Memory Layer (preference vector + decay)
    rag/               RAG knowledge base service
    providers/         Provider abstraction (base, registry, mock/, real/)
    llm/               Pluggable LLM provider interface + adapters
    tools/             Deterministic tools over the provider layer
    mcp/               Demonstrative MCP server (forecast/distance/travel_time)
    cache/             Redis price-snapshot cache + backoff/retry
    observability/     agent_runs tracing
    models/            SQLAlchemy models + Pydantic schemas
    eval/              Evaluation harness + golden scenarios
  tests/               pytest suite
  requirements.txt     pinned dependencies
  pyproject.toml       project metadata + pytest config
```

## Requirements

- **Python 3.11 or 3.12 recommended.** Some native wheels (notably `ortools`,
  `psycopg2-binary`) may not yet publish builds for the very newest Python
  releases (3.13/3.14). If `pip install` fails on a native package, create the
  virtual environment with Python 3.11/3.12.

## Setup

Windows (PowerShell):

```powershell
cd backend
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

macOS / Linux:

```bash
cd backend
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Configuration

Settings load from environment variables (and an optional `.env` file in
`backend/`). All are optional for local mock-only runs.

| Variable                | Purpose                                              | Absent ⇒                |
|-------------------------|------------------------------------------------------|-------------------------|
| `DATABASE_URL`          | Postgres connection URL                              | local default           |
| `REDIS_URL`             | Redis connection URL (price-snapshot cache)          | local default           |
| `SUPABASE_JWKS_URL`     | Supabase JWKS endpoint (RS256/ES256 verification)    | —                       |
| `SUPABASE_JWT_SECRET`   | Supabase shared HS256 secret (fallback verification) | —                       |
| `LLM_VENDOR`            | `gemini` \| `claude` \| `gpt`                         | `gemini`                |
| `AMADEUS_API_KEY` + `AMADEUS_API_SECRET` | Flights + hotels (Amadeus)          | **mock** flights/hotels |
| `OPENWEATHERMAP_API_KEY`| Weather (OpenWeatherMap)                             | **mock** weather        |
| `MAPBOX_ACCESS_TOKEN`   | Maps/routes (Mapbox)                                 | **mock** routes         |
| `TICKETMASTER_API_KEY`  | Events (Ticketmaster)                                | **mock** events         |

**Key rule:** absent credentials for a domain ⇒ that domain uses the mock
provider. Amadeus requires *both* the key and the secret to be considered
configured.

## Run

```bash
uvicorn app.main:app --reload
```

Liveness probe: `GET /health` → `{"status": "ok"}`.

## Test

```bash
cd backend
pytest
```

Verify the app imports and instantiates without the full dependency set:

```bash
python -c "import app.main; print(app.main.app.title)"
```
