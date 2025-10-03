# Scheduler Service (`services/scheduler`)

The scheduler is a lightweight polling worker that bridges the database and the solver.  It is responsible for pulling jobs out of PostgreSQL, ensuring only one runner processes a job at a time, and translating solver responses into status updates and human-readable error messages.


## Responsibilities

- Poll the `jobs` table for the oldest `QUEUED` entry.
- Atomically transition a job to `RUNNING` prior to execution.
- Materialise the solver payload by loading `J.npy`, optional vectors, and `study_request.json` from `/data/jobs/<job_id>/`.
- Carry forward annotations from the upload form—job names, target objective values, CUDA hints, and custom Optuna parameters—so downstream services see a consistent view of the request.
- Validate matrix/vector shapes while constructing the payload; malformed files surface as `FAILED` jobs with diagnostic messages.
- Invoke the solver service (`POST /run`) with the fully expanded Optuna request.
- Handle HTTP/network failures and solver-side errors, downgrading jobs to `FAILED` with diagnostic text.
- Retry the polling loop indefinitely, backing off slightly when the queue is empty.


## Container Image

- **Dockerfile**: `services/scheduler/Dockerfile`
- **Base**: `python:3.11-slim`.
- **Runtime command**: `python3 -m app.main`
- **Dependencies installed**: `psycopg2-binary`, `requests`, `numpy`, `SQLAlchemy`


## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `DATABASE_URL` | ✔ | _none_ | Connection string to the PostgreSQL instance. |
| `SOLVER_URL` | ✖ | `http://solver:8081` | Base URL of the solver service. Override when running the solver on a different host/port. |
| `SOLVER_TIMEOUT` | ✖ | unset | Float (seconds). If provided, passed to `requests.post(..., timeout=...)` to limit solver execution time. |
| `DATA_ROOT` | ✖ | `/data` | Shared volume containing `jobs/` (inputs) and `studies/` (outputs). |


## Polling Loop

`app/main.py` implements a `run_once(conn)` helper that encapsulates the main workflow:

1. `SELECT_NEXT` retrieves the oldest queued job (`ORDER BY created_at ASC LIMIT 1`).
2. If no job is found, the caller sleeps for four seconds; otherwise the loop continues after a two-second pause.
3. Before contacting the solver, `MARK_RUNNING` sets the status to `RUNNING` and commits the transaction. This prevents duplicate scheduling when multiple worker instances run concurrently.
4. The solver endpoint is called with `requests.post`. The payload contains the coupling matrix, optional vectors, study metadata (including any `study_args`/`target_objective_value`), the derived energy shift, and CUDA preferences required by `nqa/master.py`.
5. Responses are interpreted as follows:
   - `response.ok == True`: the job is marked `DONE`, the raw response body is logged (truncated to 500 characters), and the loop proceeds.
   - Non-200 responses: the body is parsed (JSON preferred) via `_parse_solver_response()` to extract meaningful details. The job is marked `FAILED` with an error message truncated to 8 KB.
   - Network/timeout errors raise a `requests.RequestException`, caught to mark the job `FAILED` with the exception text.

The outer `while True` handles database connectivity issues by sleeping four seconds before retrying the connection.


## Database Access Patterns

- The scheduler opens a long-lived connection (`psycopg2.connect`) and reuses it inside the inner polling loop to minimise overhead.
- Cursor factories use `RealDictCursor` when reading job rows so column access is dictionary-based (`job["id"]`).
- Updates are committed explicitly to guarantee another scheduler instance sees the new status immediately.


## Interaction with the Solver

- Endpoint constructed from `SOLVER_URL.rstrip('/') + '/run'`.
- Request payload: JSON document with keys `J_matrix`, optional `h_vector`/`g_vector`, and every metadata item captured by the API—`study_name` (defaults to the job ID), optional `job_name`, derived `input_format`, `energy_shift`, and any submitted `study_args` (including `target_objective_value`). Per-study hyperparameters continue to fall back to the solver defaults when unset.
- Timeout: `SOLVER_TIMEOUT` if provided; otherwise the request may block until the solver responds.
- Successful responses are logged verbatim (limited to 500 characters) to aid debugging and to capture solver stdout/stderr tails.
- Errors are serialised to strings before being written to the database. When the solver returns a JSON object with `detail`/`error`/`stdout`/`stderr` keys, the scheduler normalises the text into a compact message.


## Failure Handling

- **Database outage**: Caught at the top level; the scheduler prints the exception and retries after a four-second delay.
- **Solver unreachable**: Network errors trigger a `FAILED` status with `solver request failed: ...` stored in `jobs.error`.
- **Solver HTTP 500**: The response body is parsed and recorded. Re-running the job requires resubmission (e.g., via the API) once the root cause is fixed.


## Running Locally

1. Install dependencies: `pip install psycopg2-binary SQLAlchemy requests numpy`.
2. Set environment variables, e.g.:
   ```bash
   export DATABASE_URL=postgresql://nqa:nqa_password@localhost:5432/nqa
   export SOLVER_URL=http://localhost:8081
   export DATA_ROOT=/path/to/shared-data
   ```
3. Launch with `python -m app.main` inside `services/scheduler`.

Ensure a solver instance is reachable before starting the scheduler; otherwise every job will immediately fail.


## Observability

- Logs are printed to stdout with `[scheduler]` prefixes. Use `docker compose logs scheduler` during operations.
- When present, friendly job names are emitted alongside UUIDs in log lines to ease traceability.
- Job transitions are recorded in the database (`status`, `error`, `updated_at`).  Querying via SQL is the authoritative source for system state.
- The scheduler does not currently expose Prometheus metrics or health endpoints. When running multiple replicas, rely on container orchestrator liveness probes instead.


## Extension Points

- **Retry semantics**: For long-running jobs you can layer exponential back-off or more granular status tracking (e.g., storing solver response codes).
- **Parallelism**: The existing logic is safe for horizontal scaling—multiple scheduler instances can run concurrently as long as they share the same database.
- **Custom metadata**: Extending the API to persist additional fields in `study_request.json` (for example, CUDA device hints) will automatically flow through the scheduler—`_build_solver_payload` copies any recognised keys into the solver payload with validation.
