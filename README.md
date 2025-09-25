# Neural Quantum Annealing Platform

This repository packages a Neural Quantum Annealing (NQA) workflow into a set of containerised services.  It accepts Sherrington–Kirkpatrick style Ising instances (`J`, optional `h`/`g` vectors), schedules them through a GPU-enabled solver, and exposes a web/API surface for submitting jobs and downloading results.  All runtime state (inputs, results, metadata) is persisted inside Docker volumes so the whole system can be launched with a single `docker compose up`.

This server is designed to run on a machine with a single GPU and be easily deployable.

## High-Level Architecture

```
┌────────┐       job              ┌──────────┐        job metadata        ┌──────────┐
│  User  │ ────────────────────▶ │   API    │ ────────────────────────▶ │ Postgres │
└────────┘ ◀──────────────────── │ (FastAPI)│                            └──────────┘
              results             └────┬─────┘                              ▲     ▲
                                       │                                    │     │
                                       │                                    │     │
                                       │                                    │     │
                            ┌──────────▼──────────┐         status updates  │     │
                            │     Scheduler       │◀───────────────────────┘     │
                            │  (polling worker)   │                               │
                            └──────────┬──────────┘                               │
                                       │ POST /run                                │
                            ┌──────────▼──────────┐                               │
                            │      Solver         │             results           │            
                            │ (FastAPI + JAX NQA) │───────────────────────────────┘
                            └─────────────────────┘
```

- **API (`services/api`)**: FastAPI application with an HTML form for uploads, REST endpoints for job management, and result packaging. All user interactions—both submissions and downloads—flow through this service.
- **Scheduler (`services/scheduler`)**: Background worker that polls PostgreSQL for `QUEUED` jobs, flips them to `RUNNING`, and asks the solver to execute them.
- **Solver (`services/solver`)**: GPU-ready FastAPI service that shells into the JAX-based annealer (`nqa/worker.py`), writes artefacts to `/data/results/<job_id>`, and streams truncated logs back to the scheduler.
- **PostgreSQL (`db`)**: Tracks job metadata and error messages.
- **Shared Data Volume (`shared-data`)**: Mounted into API/Scheduler/Solver containers to exchange job inputs (`/data/jobs/…`) and outputs (`/data/results/…`).


## Repository Layout

- `docker-compose.yml` — orchestrates the database and application services, wiring environment variables and shared volumes.
- `.env` — default PostgreSQL credentials consumed by Compose; edit before deploying to production.
- `services/api` — API service sources and container definition.
- `services/scheduler` — polling worker sources.
- `services/solver` — solver service, JAX/Optax code (`nqa/`), and its container definition.
- `paper.pdf` — reference manuscript describing the underlying NQA method.
- `J.npy` — example coupling matrix useful for smoke tests.
- `shared-data/` — empty host folder mounted as the shared volume during local runs.
- `docs/` *(created in this change)* — service-focused documentation (`api.md`, `scheduler.md`, `solver.md`).


## Prerequisites

1. Docker 24+ with the Compose plugin (`docker compose`).
2. NVIDIA Container Toolkit (for GPU access inside the solver container) and a compatible driver/runtime for `nvcr.io/nvidia/jax:25.08-py3`.
3. Optional: Python 3.11 environment if you want to run components locally outside Docker.


## Quick Start

```bash
git clone <repo-url>
cd NQA/Release

# Adjust credentials, ports, or solver defaults here if necessary
cp .env.example .env   # if you keep a template, otherwise edit the existing .env

# Build images (solver image pull can take several minutes the first time)
docker compose build

# Start the full stack
docker compose up

# Visit the web UI
open http://localhost:8000
```

The solver container is exposed on `http://localhost:8081`; the scheduler reaches it via the internal Docker network (`http://solver:8081`).  PostgreSQL listens on `localhost:5432` using the credentials in `.env`.


## Configuration Reference

| Variable | Default | Used By | Purpose |
|----------|---------|---------|---------|
| `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB` | see `.env` | db, api, scheduler | Database credentials. |
| `DATA_ROOT` | `/data` | api, solver | Root of shared-data volume inside containers (`jobs/`, `results/`). |
| `SOLVER_URL` | `http://solver:8081` | scheduler | Base URL for the solver service. Override when running solver outside Compose. |
| `SOLVER_TIMEOUT` | unset (no timeout) | scheduler | Optional float (seconds) to bound solver HTTP request duration. |
| `NQA_WORKER_SCRIPT` | `/nqa/nqa/worker.py` | solver | Path to the annealer entrypoint inside the solver image. |
| `NQA_PYTHON_BIN` | `<sys.executable>` | solver | Override interpreter used to run the worker (defaults to container Python). |
| `NQA_DEFAULT_CUDA_DEVICE` | unset | solver | If defined, appended as `--cuda_device` and `CUDA_VISIBLE_DEVICES` when the job lacks explicit GPU selection. |

> For a deeper dive into each service’s configuration knobs and API surface, see the files under `docs/`.


## Job Lifecycle

1. **Submission**: Users upload a `J.npy` file (and optionally `h_vector.npy`, `g_vector.npy`, plus CLI arguments) via the HTML form or a direct `POST /upload` request.
2. **Persistence**: The API writes the payload to `/data/jobs/<job_id>/` and creates a `Job` row (`status=QUEUED`).
3. **Scheduling**: The scheduler polls every few seconds, marks jobs `RUNNING`, and POSTs `{"job_id": "…"}` to the solver’s `/run` endpoint.
4. **Execution**: The solver assembles a command for `nqa/worker.py`, ensures a result directory exists, executes it, and captures stdout/stderr tails.
5. **Completion**: On success the scheduler updates the job to `DONE`; on error it records the (truncated) failure details in `jobs.error`.
6. **Retrieval**: Users can query `GET /jobs`, `GET /jobs/{id}`, and download zipped results via `GET /jobs/{id}/download` once complete.


## API Surface (Summary)

All endpoints live under the API service (`http://localhost:8000` by default).  Full parameter descriptions and payload schemas are documented in `docs/api.md`.

| Method & Path | Description |
|---------------|-------------|
| `GET /` | HTML upload form with dynamic solver options. |
| `POST /upload` | Multipart submission. Returns `{"job_id": "…", "status": "QUEUED"}`. |
| `GET /jobs` | List jobs (newest first). |
| `GET /jobs/{job_id}` | Single job with timestamps and error message. |
| `GET /jobs/{job_id}/download` | Zip download of `/data/results/<job_id>` (requires status `DONE`). |

The solver exposes `GET /health` and `POST /run` internally; details are in `docs/solver.md`.


## Data Layout

The shared Docker volume is mounted at `/data` for application containers and corresponds to `shared-data/` on the host.  The directory structure per job looks like this:

```
/data
 ├── jobs
 │   └── <job_id>
 │        ├── J.npy
 │        ├── h_vector.npy          # optional
 │        ├── g_vector.npy          # optional
 │        └── cli_args.json         # optional solver arguments (non-defaults only)
 └── results
     └── <job_id>
          ├── results.txt
          ├── data.npz
          └── plots/
```

When defining `--save_path` in custom CLI parameters, ensure it remains inside `/data/results/<job_id>` so the ZIP download endpoint can locate the artefacts.


## Operations & Monitoring

- **Logs**: Each service logs to stdout. Use `docker compose logs <service>` for inspection. The solver returns truncated stdout/stderr in its HTTP response; the scheduler logs both the response snippet and any error message saved to the database.
- **Database access**: `docker compose exec db psql -U $POSTGRES_USER $POSTGRES_DB` gives a psql shell. The `jobs` table records the latest status/error.
- **Health check**: `curl http://localhost:8081/health` verifies the solver service; API responds to `/` and `/jobs` even before jobs exist.
- **Cleaning state**: `docker compose down -v` drops the PostgreSQL and shared-data volumes (irreversible). Alternatively delete directories inside `shared-data/` to reset inputs/results.


## Development

- **Running services individually**:
  - API: `uvicorn app.main:app --reload --port 8000` (set `DATABASE_URL` and `DATA_ROOT` first).
  - Solver: `uvicorn app.main:app --reload --port 8081` (provide `DATA_ROOT`, `NQA_WORKER_SCRIPT`, and GPU access if needed).
  - Scheduler: `python -m app.main` (requires DB connection and a reachable solver URL).
- **Editing solver code**: The `nqa/` directory contains the JAX implementation, utilities, and tests. Keep modifications backwards compatible with `nqa/worker.py` CLI arguments.
- **Local testing**: Inside `services/solver/nqa`, run `pytest -q`.  GPU tests assume CUDA availability; CPU-only runs may require setting `XLA_PYTHON_CLIENT_PREALLOCATE=false` or similar flags.
- **Linting/formatting**: Not enforced by scripts in this repo; adopt `ruff`, `black`, etc., if you integrate into CI.


## Troubleshooting

- **Solver exits with code >0**: The scheduler marks the job FAILED and stores truncated stderr text in `jobs.error`. Inspect `/data/results/<job_id>` for full logs. Check that CLI arguments are valid (e.g., float vs. complex strings).
- **GPU unavailable**: Make sure `nvidia-smi` works on the host and the Docker daemon is configured for GPU passthrough. Optionally set `NQA_DEFAULT_CUDA_DEVICE=0` so the solver explicitly binds to GPU 0.
- **Upload rejected**: Only `.npy` files are accepted for matrices/vectors. The API ensures `h_vector`/`g_vector` aren’t combined with their scalar counterparts.
- **Timeouts**: Set `SOLVER_TIMEOUT` to guard against long-running jobs; FAILED jobs can be resubmitted once parameters are tuned.
- **Legacy Spark runner**: A Spark-based executor (`services/scheduler/app/spark_job.py`) remains in the tree for reference but is not used by the current Docker Compose stack.


## Additional Documentation

- `docs/api.md` — deep dive into the API service, database schema, and REST contract.
- `docs/scheduler.md` — scheduler internals, polling strategy, and failure handling.
- `docs/solver.md` — solver container layout, execution pipeline, and CUDA considerations.

Contributions and suggestions are welcome—open an issue or submit a PR with proposed changes.
