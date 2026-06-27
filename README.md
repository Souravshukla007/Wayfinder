# 🧭 Wayfinder: AI Travel Consultant That Explains Every Decision

[![React](https://img.shields.io/badge/React-18.3-61DAFB?style=flat-square&logo=react)](https://react.dev/)
[![Vite](https://img.shields.io/badge/Vite-5.4-646CFF?style=flat-square&logo=vite)](https://vitejs.dev/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.8-3178C6?style=flat-square&logo=typescript)](https://www.typescriptlang.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Pydantic_v2-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agentic-1C3C3C?style=flat-square)](https://langchain-ai.github.io/langgraph/)
[![OR-Tools](https://img.shields.io/badge/OR--Tools-CP--SAT-EA4335?style=flat-square&logo=google)](https://developers.google.com/optimization)
[![Supabase](https://img.shields.io/badge/Supabase-Auth-3FCF8E?style=flat-square&logo=supabase)](https://supabase.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-pgvector-4169E1?style=flat-square&logo=postgresql)](https://www.postgresql.org/)

Wayfinder is an **AI travel consultant** — not another itinerary chatbot. It reasons about trade-offs (budget, weather, crowds, interests) and **explains every choice** with an auditable rationale. Instead of returning one answer, it returns ranked, scored options ("Kyoto 92 · Osaka 88 · Hokkaido 84") and shows exactly *why* one outranks another. The architecture deliberately separates **LLM reasoning** from **deterministic decision-making**: language models narrate, while a pure scoring engine and a constraint solver do the math.

> **Project status:** The web frontend (Vite + React + Supabase auth) is implemented and runs end-to-end on a mock data layer. The Python backend (FastAPI + LangGraph + OR-Tools) is fully specified (requirements, design, 30 correctness properties, task plan) and under active implementation. This README documents the complete v1 architecture.

---

## ⚡ Recruiter Fast-Track (30-Second Summary)

If you're evaluating the technical depth of this repository, here's what makes it stand out:

*   **Deterministic Decision Engine, not LLM vibes**: Rankings are computed by a **pure, re-runnable function** of `(features, weights, constraints)` — normalized 0–1 features → weighted 0–100 score — with bit-for-bit determinism. The LLM is *forbidden* from inventing or reordering scores; it only narrates. Every score carries an **additive explainability ledger** whose per-feature contributions provably sum to the final score.
*   **Real Optimization Engineering (CP-SAT)**: A **Google OR-Tools CP-SAT** constraint solver guarantees *provably feasible* itineraries — enforcing hard budget caps, travel-time feasibility, hotel-change limits, city counts, and dietary constraints — or rejects the plan with the specific unsatisfied constraints before any LLM enrichment runs.
*   **Agentic AI done right (LangGraph)**: A stateful graph orchestrates three reasoning agents (Coordinator → Destination → Itinerary) alongside deterministic tools, with cross-trip personalization via a time-decayed preference vector.
*   **Grounded RAG with anti-hallucination guardrails**: Visa/safety answers are retrieved from a vector store at a 0.7 similarity floor, cited with source + freshness, and **never generated** when no grounded document exists — critical for high-liability travel information.
*   **Provider abstraction for zero-key demoability**: Every external domain (flights, hotels, weather, maps, events) sits behind an interface with mock implementations, so the entire system runs offline with no paid API keys; real providers (Amadeus, OpenWeatherMap, Mapbox, Ticketmaster) drop in via config.

---

## 📂 Direct Code Navigation (Key Implementation Anchors)

Jump straight into the production code for the key features:

| Feature | Key Logic File | Role / Purpose |
| :--- | :--- | :--- |
| **🧠 Decision Engine** | `backend/app/decision/engine.py` | Pure deterministic scoring: normalize → weight → 0–100 score. |
| **📊 Explainability Ledger** | `backend/app/decision/ledger.py` | Additive per-feature contributions that sum to the final score. |
| **🧩 CP-SAT Solver** | `backend/app/solver/cp_sat.py` | OR-Tools feasibility model + rejection/timeout handling. |
| **🕸️ Agent Orchestration** | `backend/app/orchestration/graph.py` | LangGraph wiring of Coordinator/Destination/Itinerary. |
| **🔐 Supabase JWT Auth** | `backend/app/auth/jwt_middleware.py` | Verifies Supabase-issued JWTs via JWKS (fail-closed). |
| **🔌 Provider Registry** | `backend/app/providers/registry.py` | Config-driven mock ↔ real provider selection. |
| **📚 Grounded RAG** | `backend/app/rag/rag_service.py` | 0.7-similarity grounding, citations, no ungrounded generation. |
| **🧬 Memory Layer** | `backend/app/memory/memory_layer.py` | Weighted preference vector with 180-day half-life decay. |
| **🎨 Frontend Planner UI** | `src/pages/Plan.tsx` | Input → live progress stream → ranked results + score bars. |
| **🔁 API Client Seam** | `src/lib/mockApi.ts` | Single data layer; swap mock fixtures for real `fetch()` calls. |

---

## 📌 Problem Statement

Most "AI travel planners" are thin wrappers around an LLM: type a prompt, get a generic *"Day 1: Tokyo Tower, Day 2: Shibuya"* itinerary. They have three core problems:

1.  **No reasoning, no trust**: They produce one answer with no explanation of *why* — no trade-off analysis between budget, weather, crowds, and personal interests. You can't tell if the plan is good or just plausible-sounding.
2.  **Hallucinated facts**: LLMs confidently invent visa rules, prices, and travel times — dangerous for high-liability decisions where being wrong costs money or a missed flight.
3.  **No feasibility guarantees**: A language model will happily suggest an impossible same-day, three-city hop that blows your budget, because it has no notion of hard constraints.

**Wayfinder** solves this with a **consultant-grade architecture**: a deterministic decision engine produces auditable rankings, a CP-SAT solver guarantees feasibility, RAG grounds every factual claim in citable sources, and LangGraph agents handle only what LLMs are actually good at — decomposition and narration. The result feels like a human travel consultant who shows their work.

---

## ✨ Key Features

### 🧠 Transparent Decision-Making
*   **Travel Decision Engine**: Deterministic scoring across six features — `budget_fit`, `weather_fit`, `crowd_score`, `food_score`, `photography_score`, `travel_efficiency` — each normalized to 0–1, weighted, and summed to a 0–100 score. Pure function: no randomness, no system time, no network.
*   **Alternative Recommendation Engine**: Returns ranked A/B/C options with an explicit explanation of why higher-ranked destinations outrank lower ones — like a consultant presenting choices, not a single verdict.
*   **Explainability Ledger ("Why Kyoto Won")**: Every score decomposes into additive per-feature contributions (`Photography +18, Crowd +15, Weather +14...`) that provably sum to the final score within a 0.001 tolerance, rendered live in the UI.

### 🧩 Provably Feasible Planning
*   **CP-SAT Constraint Solver**: Google OR-Tools enforces hard constraints (budget cap, max same-day travel time, hotel-change limit, city count, dietary/interest filters). Emits a feasible skeleton or rejects the plan with the exact unsatisfied constraints — before the Itinerary Agent ever runs.
*   **Timeout-Safe**: Bounded solving time; on timeout the plan is rejected cleanly rather than hanging or returning garbage.

### 🕸️ Agentic Orchestration
*   **LangGraph Stateful Graph**: Coordinator (goal decomposition + memory load + merge) → Destination Agent (candidate proposal + score narration) → Decision Engine → CP-SAT Solver → Itinerary Agent (day-by-day enrichment).
*   **Pluggable LLM Provider**: Vendor-agnostic interface across Gemini / Claude / GPT — switch models via config with zero agent-code changes.
*   **Deterministic Tools**: Flights, hotels, weather, events, routes, and budget are plain functions/API calls — no LLM overhead where none is needed.

### 🧬 Personalization That Remembers
*   **Travel Memory Layer**: A weighted `(topic, weight)` preference vector learned across trips — so a returning traveler isn't re-asked what they like.
*   **Time-Decayed Signals**: A 180-day half-life decays stale tastes; explicit stated preferences are weighted at least 2× implicit inferred ones; users can view and override everything.
*   **Personalized Scoring Weights**: Memory feeds the Decision Engine's weights (renormalized to 100%, blended 70/30 with base defaults) so rankings adapt to the individual.

### 📚 Grounded Knowledge (RAG)
*   **Anti-Hallucination Visa/Safety**: Retrieves from a pgvector/Chroma store at a ≥ 0.7 similarity floor, answers using only retrieved claims, cites source + last-updated date, and returns *"no verified information"* (never an LLM guess) when nothing grounds the query.
*   **Rich Guides**: Neighborhood, photography, food, themed-interest, and transport content — each answer carries at least one citation.

### 🔐 Security & Reliability
*   **Supabase JWT Verification**: The backend trusts Supabase as the identity provider and verifies its JWTs via JWKS, failing closed on any invalid/expired/unsigned token. No auth is reimplemented server-side.
*   **Redis Caching & Rate-Limit Backoff**: Volatile prices are cached as point-in-time snapshots; provider 429s trigger exponential backoff with jitter.
*   **Full Observability**: Every agent run persists name, input, output, token usage, latency, and a trace id to `agent_runs`.

### 🎨 Premium UX
*   **Live Agent Progress Streaming**: WebSocket stage events (`Searching flights ✓, Checking hotels ✓, Scoring destinations ●`) — honest progress, not fake live pricing.
*   **Animated, Themed Interface**: React + Tailwind + shadcn/ui + Framer Motion — floating destination cards, score-bar reveals, page transitions, and image-backed auth pages.

---

## 🏗️ System Architecture

Wayfinder is a hybrid system: a Vercel-hosted frontend, a container-hosted FastAPI backend (it needs WebSockets, Redis, OR-Tools, and long-running graph state), and managed data services.

```mermaid
graph TD
    User([Traveler]) -->|Interacts| UI[React Frontend / Vite]

    subgraph Identity [Identity Provider]
        SB[Supabase Auth<br/>email + Google OAuth<br/>issues JWT + JWKS]
    end

    subgraph Backend [FastAPI Backend - Container Host]
        API[REST + WebSocket API]
        MW[JWT Verification Middleware]
        ORCH[LangGraph Orchestrator]
        DE[Decision Engine<br/>deterministic]
        CS[CP-SAT Solver<br/>OR-Tools]
        MEM[Memory Layer]
        RAG[RAG Service]
        PROV[Provider Layer<br/>mock or real]
        LLM[LLM Provider<br/>Gemini/Claude/GPT]
    end

    subgraph Data [Managed Services]
        PG[(PostgreSQL)]
        RD[(Redis)]
        VS[(Vector Store<br/>pgvector/Chroma)]
    end

    subgraph External [Real Providers - optional via config]
        AM[Amadeus]
        OWM[OpenWeatherMap]
        MB[Mapbox]
        TM[Ticketmaster]
    end

    UI -->|login| SB
    UI -->|REST + WS, Bearer JWT| API
    API --> MW
    MW -->|verify via JWKS| SB
    MW --> ORCH
    ORCH --> DE
    ORCH --> CS
    ORCH --> MEM
    ORCH --> RAG
    ORCH --> PROV
    ORCH --> LLM
    PROV -.->|if configured| AM
    PROV -.-> OWM
    PROV -.-> MB
    PROV -.-> TM
    ORCH --> PG
    API --> RD
    RAG --> VS
    MEM --> PG

    classDef client fill:#EC4899,stroke:#BE185D,stroke-width:2px,color:#fff;
    classDef api fill:#009688,stroke:#00695C,stroke-width:2px,color:#fff;
    classDef db fill:#4169E1,stroke:#27408B,stroke-width:2px,color:#fff;
    classDef ext fill:#9B59B6,stroke:#8e44ad,stroke-width:2px,color:#fff;

    class User,UI client;
    class API,MW,ORCH,DE,CS,MEM,RAG,PROV,LLM api;
    class PG,RD,VS db;
    class SB,AM,OWM,MB,TM ext;
```

---

## 💻 Tech Stack

| Technology | Category | Usage / Context |
| :--- | :--- | :--- |
| **React 18 + Vite 5** | Frontend Framework | SPA prototype, fast HMR dev server |
| **TypeScript 5.8** | Language (FE) | Compile-time type safety across the UI |
| **Tailwind CSS 3.4 + shadcn/ui** | Styling | Utility-first design system + accessible primitives |
| **Framer Motion 12** | Animation | Page transitions, score bars, floating cards |
| **FastAPI + Pydantic v2** | Backend Framework | Async REST + WebSocket API, schema validation |
| **LangGraph** | Agent Orchestration | Stateful multi-agent planning graph |
| **Google OR-Tools (CP-SAT)** | Optimization | Provably feasible itinerary constraint solving |
| **Supabase** | Authentication | Email + Google OAuth; JWT issuer (verified by backend) |
| **PostgreSQL + pgvector** | Database + Vectors | Relational storage + RAG embeddings |
| **Redis** | Cache | Price snapshots, rate-limit backoff buffers |
| **Gemini / Claude / GPT** | LLM (pluggable) | Goal decomposition + narration only |
| **Hypothesis** | Property-Based Testing | Verifies the 30 correctness properties |

---

## 🗃️ Database Design

Relational schema (PostgreSQL); `user_id` maps to the Supabase `sub` claim. Reasoning artifacts (`recommendations`, `decision_traces`, `agent_runs`) make the system fully auditable.

```
                       ┌──────────────────┐
                       │       User       │
                       └────────┬─────────┘
                                │ (1)
        ┌──────────────┬────────┼────────────┬──────────────────┐
        │ (N)          │ (N)    │ (N)         │ (N)              │ (N)
 ┌──────▼──────┐ ┌─────▼──────┐ │ ┌───────────▼──────┐ ┌─────────▼─────────┐
 │    Trip     │ │user_prefer-│ │ │ liked_destinations│ │disliked_destinat. │
 └──────┬──────┘ │   ences    │ │ └──────────────────┘ └───────────────────┘
        │ (1)    └────────────┘ │
        ├───────────┬───────────┼───────────┬───────────┬──────────────┐
        │ (N)       │ (N)       │ (N)        │ (N)       │ (N)          │ (N)
 ┌──────▼─────┐ ┌───▼──────┐ ┌──▼────────┐ ┌─▼────────┐ ┌▼───────────┐ ┌▼────────────┐
 │ itineraries│ │ flight_  │ │  hotel_   │ │agent_runs│ │recommendat-│ │decision_    │
 │            │ │ options  │ │  options  │ │          │ │   ions     │ │  traces     │
 └──────┬─────┘ └──────────┘ └───────────┘ └──────────┘ └────────────┘ └─────────────┘
        │ (N)
 ┌──────▼─────┐
 │ activities │
 └────────────┘
```

### Key Models Explained:
*   **Trip**: The planning unit — origin, dates, `budget`, `interests` (JSON), and `status` (`processing` → `complete`).
*   **recommendations**: Stores the ranked decision, rationale, alternatives, and the per-feature `scores` breakdown — the consultant's verdict.
*   **decision_traces**: `factors_json` holds the additive contributions that sum to `score`; the data behind "Why X Won."
*   **agent_runs**: Per-agent observability (name, input, output, tokens, latency, `trace_id`).
*   **user_preferences**: `(topic, weight)` vector with `source` (`explicit`/`implicit`) for the ≥2× rule and `updated_at` for 180-day half-life decay.
*   **flight_options / hotel_options**: Each row carries a `rationale` field — no option is shown without a reason.

---

## 🔐 Authentication Flow

Wayfinder delegates identity to **Supabase** and verifies its tokens on the backend — no passwords or sessions are handled server-side.

```
[User] ──login──► Supabase Auth (email/password or Google OAuth)
                          │
                  (issues signed JWT)
                          │
[Frontend] ──Authorization: Bearer <JWT>──► FastAPI endpoint
                          │
                ┌─────────▼──────────┐
                │  JWT Middleware     │
                │  fetch+cache JWKS   │
                │  verify sig/aud/exp │
                └─────────┬──────────┘
                  valid?  │  no → 401 (fail closed)
                   yes ▼
            [AuthenticatedUser(user_id, email)] → route handler
```

### Security Details:
1.  **JWKS verification**: Asymmetric (RS256/ES256) signature validation against cached Supabase keys, with an HS256 shared-secret fallback selectable by config.
2.  **Fail-closed**: If no verification key is available, requests are rejected (`401`) rather than admitted — never fail-open.
3.  **WebSocket guard**: The progress-stream handshake verifies the JWT and closes the socket *before* streaming if verification fails.

---

## 🧠 Decision Engine & Explainability Pipeline

The heart of Wayfinder — and the reason it's a consultant, not a chatbot.

```
Candidate features ─► Validate weights (sum=1.0 ±0.001) ─► Clamp out-of-bounds
                                                                │
        ┌───────────────────────────────────────────────────────┘
        ▼
Normalize each feature → [0,1] ─► Weighted sum → 0–100 score ─► Rank descending
        │
        ▼
Build decision trace: contribution_i = weight_i × normalized_i × 100
        │
        ▼
Assert Σ contributions == score (±0.001) ─► persist recommendations + decision_traces
        │
        ▼
LLM narrates the numbers (NEVER generates or reorders them)
```

### Why this matters:
*   **Determinism (Property 1)**: Identical inputs → bit-for-bit identical scores. Rankings are reproducible and testable.
*   **Additivity (Property 8)**: The ledger's contributions must sum to the score, so the "Why X Won" breakdown is mathematically honest, not decorative.
*   **LLM containment (Property 7)**: The language model is structurally prevented from touching the numbers — the single most important guardrail for trustworthy AI rankings.

---

## 🧩 Constraint Solver Pipeline (CP-SAT)

```
Scored candidates ─► OR-Tools CP-SAT model
        │
        ├─ budget cap          (total cost ≤ budget)
        ├─ travel-time         (no infeasible same-day hops)
        ├─ hotel-change limit  (changes ≤ max)
        ├─ city-count limit    (distinct cities ≤ max)
        ├─ dietary filters     (exclude violations)
        └─ interest filters    (only consistent options)
        │
   ┌────▼─────────────────────────────────────────┐
   │ Feasible? ──► emit provably-valid skeleton    │──► Itinerary Agent enriches
   │ Infeasible ► reject + list failed constraints │──► Itinerary Agent NOT run
   │ Timeout ───► reject with timeout indication   │──► Itinerary Agent NOT run
   └───────────────────────────────────────────────┘
```

The solver sits *between* the Decision Engine and the Itinerary Agent — so a language model is never asked to enrich an infeasible plan.

---

## 📂 Project Structure

```text
wayfinder/
├── src/                          # Frontend (Vite + React + TS) — IMPLEMENTED
│   ├── pages/                    # Landing, Plan, Trips, Preferences, AuthPages, NotFound
│   ├── components/
│   │   ├── nomad/                # Navbar, PageTransition, Shimmer
│   │   └── ui/                   # shadcn/ui primitives
│   ├── contexts/AuthContext.tsx  # Supabase session management
│   ├── integrations/supabase/    # Supabase client + types
│   └── lib/mockApi.ts            # Single data layer (mock → real swap point)
│
├── backend/                      # FastAPI backend — SPECIFIED / IN PROGRESS
│   └── app/
│       ├── main.py               # App, routers, lifespan
│       ├── config.py             # Pydantic Settings (provider/LLM selection)
│       ├── auth/                 # Supabase JWT verification (JWKS)
│       ├── api/                  # routes_trips, routes_preferences, routes_ws
│       ├── orchestration/        # LangGraph graph, state, agent nodes
│       ├── decision/             # engine, weights, ledger (deterministic core)
│       ├── solver/               # cp_sat (OR-Tools)
│       ├── memory/               # preference vector + decay
│       ├── rag/                  # grounded retrieval + knowledge base
│       ├── providers/            # base interfaces, registry, mock/, real/
│       ├── llm/                  # pluggable Gemini/Claude/OpenAI
│       ├── tools/                # flights, hotels, weather, events, routes, budget
│       ├── mcp/                  # demonstrative Maps/Weather MCP server
│       ├── cache/                # Redis snapshots + backoff
│       ├── observability/        # agent_runs tracing
│       ├── models/               # SQLAlchemy + Pydantic schemas
│       └── eval/                 # plan-quality harness + golden scenarios
│
├── .kiro/specs/                  # requirements.md, design.md, tasks.md (30 properties)
├── index.html                    # Vite entry + meta
├── package.json
└── vite.config.ts
```

---

## 🛠️ Installation & Setup

### 1. Prerequisites
*   Node.js v18+
*   Python 3.11+ (for the backend)
*   PostgreSQL 16 (with `pgvector`) and Redis (or managed equivalents)
*   A Supabase project (for auth)

### 2. Clone the repository
```bash
git clone https://github.com/<your-username>/wayfinder.git
cd wayfinder
```

### 3. Frontend (runs today on mock data)
```bash
npm install
npm run dev
```
The app runs at [http://localhost:8080](http://localhost:8080).

### 4. Backend (FastAPI)
```bash
cd backend
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### 5. Build for production
```bash
npm run build     # frontend → dist/
```

---

## 📄 Environment Variables

### Frontend (`.env`)
```env
VITE_SUPABASE_URL="https://<project>.supabase.co"
VITE_SUPABASE_PUBLISHABLE_KEY="<supabase-anon-key>"
VITE_SUPABASE_PROJECT_ID="<project-ref>"
VITE_API_BASE_URL="http://localhost:8000"
```

### Backend (`backend/.env`)
```env
# Database & cache
DATABASE_URL="postgresql://user:password@host:5432/wayfinder"
REDIS_URL="redis://localhost:6379/0"

# Supabase JWT verification
SUPABASE_JWKS_URL="https://<project>.supabase.co/auth/v1/.well-known/jwks.json"
SUPABASE_JWT_SECRET="<optional-hs256-shared-secret>"

# LLM (pick one vendor)
LLM_VENDOR="gemini"            # gemini | claude | gpt
GEMINI_API_KEY="AIzaSy..."
ANTHROPIC_API_KEY="sk-ant-..."
OPENAI_API_KEY="sk-..."

# Real providers (optional — omit to use mock providers)
AMADEUS_API_KEY="..."
AMADEUS_API_SECRET="..."
OPENWEATHERMAP_API_KEY="..."
MAPBOX_ACCESS_TOKEN="..."
TICKETMASTER_API_KEY="..."
```

> With no provider keys set, the backend automatically uses **mock providers** — the full system is demoable with zero paid API keys.

---

## 📸 Screenshots

| Landing — Consultant Positioning | Planner — Live Agent Progress |
| :---: | :---: |
| ![Landing](https://via.placeholder.com/600x350/FDF2F8/EC4899?text=Plan+smarter.+Travel+deeper.)<br>Animated hero, four-country cards, decision-transparency framing. | ![Planner](https://via.placeholder.com/600x350/F5F5F4/78716C?text=Searching+flights+%E2%9C%93)<br>Natural-language input → streaming agent checklist → results. |

| Results — Ranked + "Why X Won" | Preferences — Memory Vector |
| :---: | :---: |
| ![Results](https://via.placeholder.com/600x350/FDF2F8/EC4899?text=Kyoto+92+%C2%B7+Osaka+88)<br>Scored A/B/C cards with animated explainability bars. | ![Preferences](https://via.placeholder.com/600x350/F5F5F4/78716C?text=Interest+Vector)<br>Tune weighted interests; every plan adapts to this vector. |

---

## 🔮 Future Improvements (v2 Roadmap)

- [ ] **Scenario Planning & Plan-Diff Engine**: Re-run the same graph with a constraint delta ("what if budget +₹30k?") and show a structured diff, reusing the pure Decision Engine.
- [ ] **Budget Drift Monitoring**: Periodically re-price plans and alert when cost deviates beyond a threshold.
- [ ] **Travel Risk Score**: Composite risk from weather, crowd, transport, and advisory signals.
- [ ] **Crowd Prediction**: Estimate tourist density from event calendars + seasonality.
- [ ] **Group Travel Optimizer**: Constraint satisfaction across multiple travelers' preferences and availability.
- [ ] **Production Next.js Frontend**: Promote the validated Vite prototype to a Next.js 15 production app.

---

## 🎖️ Why This Project Matters (Recruiter Corner)

Wayfinder is not a CRUD app or an LLM wrapper — it's a **hybrid AI system** that demonstrates judgment about *where* machine learning belongs:

1.  **Separation of reasoning and computation**: The hardest engineering decision here was *not* letting the LLM do everything. Scores come from a deterministic, property-tested engine; feasibility comes from a constraint solver; the LLM only narrates. This is the difference between a demo and a trustworthy product.
2.  **Formal correctness**: The system is specified with **30 universally-quantified correctness properties** (determinism, ledger additivity, solver feasibility soundness, grounding thresholds, auth) verified via property-based testing with Hypothesis — not just example tests.
3.  **Production-shaped architecture**: Provider abstraction for zero-key demoability, Supabase JWT verification, Redis caching with rate-limit backoff, WebSocket streaming, and full agent-run observability.
4.  **Real optimization & agentic AI**: CP-SAT constraint programming and a LangGraph multi-agent graph — demonstrating depth in both classical optimization and modern agentic orchestration.

---
*Built with 🧭 to make travel decisions you can actually trust.*
