# CLAUDE.md — Indie Insights Analytics API

## Project Overview

FastAPI backend that processes DistroKid music sales exports and generates analytics insights for indie musicians. Accepts TSV/XLSX/XLS/CSV uploads of DistroKid "Excruciating Detail" exports and returns comprehensive revenue, streaming, and market analysis.

**Live domain**: farraranalytics.com
**Deployed on**: Railway (Docker-based)

## Tech Stack

- **Python 3.11** with **FastAPI 0.109** (async)
- **Pandas 2.2 / NumPy 1.26** for data processing
- **Supabase** (PostgreSQL) for persistence
- **Uvicorn** ASGI server (multi-worker)
- **Docker** for containerization

## Project Structure

```
app/
├── __init__.py              # Package init
├── main.py                  # FastAPI app, routes, middleware, file handling
├── analytics_engine.py      # DistroKidAnalyzer class — all analytics logic
└── database.py              # Supabase client, CRUD for saved analyses
Dockerfile                   # Python 3.11-slim, multi-worker Uvicorn
railway.toml                 # Railway deployment config
requirements.txt             # Python dependencies
```

## API Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/` | API info |
| GET | `/health` | Health check (returns DB enabled status) |
| POST | `/upload` | Upload DistroKid file for analysis |
| GET | `/analyses/{user_id}` | List saved analyses (metadata) |
| GET | `/analyses/{user_id}/{analysis_id}` | Get full analysis |
| DELETE | `/analyses/{user_id}/{analysis_id}` | Delete analysis |

The `/upload` endpoint accepts a `file` (multipart) and optional `user_id` string. Supported formats: `.tsv`, `.xlsx`, `.xls`, `.csv`. Max file size: 10 MB (configurable via `MAX_FILE_SIZE_MB`).

Required columns in uploaded files: `Sale Month`, `Store`, `Title`, `Quantity`, `Earnings (USD)`, `Country of Sale`.

## Environment Variables

```
SUPABASE_URL            # Supabase project URL (required for DB features)
SUPABASE_SERVICE_KEY    # Supabase service role key (required for DB features)
CORS_ORIGINS            # Comma-separated allowed origins (default: localhost:3000, localhost:4321, farraranalytics.com)
MAX_FILE_SIZE_MB        # Upload limit in MB (default: 10)
PORT                    # Server port (default: 8000)
WEB_CONCURRENCY         # Uvicorn worker count (default: 2)
```

Database features are optional — the API functions without Supabase credentials but won't persist analyses.

## Running Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run the server
uvicorn app.main:app --reload --port 8000
```

Or via Docker:
```bash
docker build -t indie-insights-api .
docker run -p 8000:8000 --env-file .env indie-insights-api
```

## Architecture & Key Patterns

### Async with thread pool offloading
All endpoints are async. CPU-bound work (pandas file parsing, analytics computation) is offloaded to a thread pool via `asyncio.get_event_loop().run_in_executor()` to avoid blocking the event loop and causing health check timeouts.

### Lazy database imports
The database module is imported lazily via `get_db()` in `main.py` — the app starts and serves requests even without Supabase credentials configured.

### Supabase client singleton
`database.py` caches the Supabase client in a module-level `_client` variable for connection reuse.

### Analytics engine
`DistroKidAnalyzer` is a class instantiated with a pandas DataFrame. It produces 11 analysis sections via `get_full_analysis()`: overview, monthly/yearly trends, song/platform/country breakdowns, concentration metrics, growth analysis, platform-song matrix, high-value markets, and catalog-excluding-top-song analysis.

### Error handling
- HTTP exceptions with appropriate status codes (400, 404, 413, 500, 503)
- Database failures during upload don't fail the upload — analysis still returns
- Structured logging at INFO/WARNING/ERROR levels

## Code Conventions

- **Type hints** on all function signatures
- **Docstrings** on modules and public functions
- **Numeric precision**: 2 decimals for USD, 4 for per-stream rates, 1 for percentages
- **NaN/Inf handling**: Always `.fillna()` and replace infinities before JSON serialization
- **Logging**: Use `logging.getLogger(__name__)` per module, structured messages

## Testing

No automated test suite exists yet. The API is tested manually via file uploads.

## Deployment

Deployed to Railway using Docker. The `railway.toml` configures:
- Health check on `/health` with 300s timeout
- Restart on failure (max 3 retries)
- Dockerfile-based build

## Common Tasks for AI Assistants

- **Adding a new analysis**: Add a method to `DistroKidAnalyzer` in `analytics_engine.py`, then include it in `get_full_analysis()`.
- **Adding a new endpoint**: Add the route in `main.py` following existing patterns (async, error handling, logging).
- **Modifying the database schema**: Update `database.py` functions and ensure the Supabase table matches.
- **Changing CORS or config**: Modify environment variable handling in `main.py`.
