# Scheduler Service (`services/scheduler`)

The scheduler is a lightweight polling worker that bridges the database and the solver.  It is responsible for pulling jobs out of PostgreSQL, ensuring only one runner processes a job at a time, and translating solver responses into status updates and human-readable error messages.


## Responsibilities

- Poll the `jobs` table for the oldest `QUEUED` entry.
- Atomically transition a job to `RUNNING` prior to execution.
- Invoke the solver service (`POST /run`) with the job identifier.
- Handle HTTP/network failures and solver-side errors, downgrading jobs to `FAILED` with diagnostic text.
- Retry the polling loop indefinitely, backing off slightly when the queue is empty.


## Container Image

- **Dockerfile**: `services/scheduler/Dockerfile`
- **Base**: `bitnami/spark:3.5` (currently used only for its Python + Spark environment; Spark execution is legacy).
- **Runtime command**: `python3 -m app.main`
- **Dependencies installed**: `psycopg2-binary`, `SQLAlchemy`, `requests`

> The historical Spark-based executor (`app/spark_job.py`) is still present for archival purposes but is not imported or executed by the current scheduler.


## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `DATABASE_URL` | ✔ | _none_ | Connection string to the PostgreSQL instance. |
| `SOLVER_URL` | ✖ | `http://solver:8081` | Base URL of the solver service. Override when running the solver on a different host/port. |
| `SOLVER_TIMEOUT` | ✖ | unset | Float (seconds). If provided, passed to `requests.post(..., timeout=...)` to limit solver execution time. |


## Polling Loop

`app/main.py` implements a `run_once(conn)` helper that encapsulates the main workflow:

1. `SELECT_NEXT` retrieves the oldest queued job (`ORDER BY created_at ASC LIMIT 1`).
2. If no job is found, the caller sleeps for four seconds; otherwise the loop continues immediately after the solver call.
3. Before contacting the solver, `MARK_RUNNING` sets the status to `RUNNING` and commits the transaction. This prevents duplicate scheduling when multiple worker instances run concurrently.
4. The solver endpoint is called with `requests.post`. The payload is a JSON object containing only the `job_id`.
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
- Request payload: `{ "job_id": <uuid str> }`.
- Timeout: `SOLVER_TIMEOUT` if provided; otherwise the request may block until the solver responds.
- Successful responses are logged verbatim (limited to 500 characters) to aid debugging and to capture solver stdout/stderr tails.
- Errors are serialised to strings before being written to the database. When the solver returns a JSON object with `detail`/`error`/`stdout`/`stderr` keys, the scheduler normalises the text into a compact message.


## Failure Handling

- **Database outage**: Caught at the top level; the scheduler prints the exception and retries after a four-second delay.
- **Solver unreachable**: Network errors trigger a `FAILED` status with `solver request failed: ...` stored in `jobs.error`.
- **Solver HTTP 500**: The response body is parsed and recorded. Re-running the job requires resubmission (e.g., via the API) once the root cause is fixed.


## Running Locally

1. Install dependencies: `pip install psycopg2-binary SQLAlchemy requests`.
2. Set environment variables, e.g.:
   ```bash
   export DATABASE_URL=postgresql://nqa:nqa_password@localhost:5432/nqa
   export SOLVER_URL=http://localhost:8081
   ```
3. Launch with `python -m app.main` inside `services/scheduler`.

Ensure a solver instance is reachable before starting the scheduler; otherwise every job will immediately fail.


## Observability

- Logs are printed to stdout with `[scheduler]` prefixes. Use `docker compose logs scheduler` during operations.
- Job transitions are recorded in the database (`status`, `error`, `updated_at`).  Querying via SQL is the authoritative source for system state.
- The scheduler does not currently expose Prometheus metrics or health endpoints. When running multiple replicas, rely on container orchestrator liveness probes instead.


## Extension Points

- **Retry semantics**: For long-running jobs you can layer exponential back-off or more granular status tracking (e.g., storing solver response codes).
- **Parallelism**: The existing logic is safe for horizontal scaling—multiple scheduler instances can run concurrently as long as they share the same database.
- **Spark integration**: If you plan to revive Spark-based execution, refer to `app/spark_job.py` for the previous approach (submitting `spark-submit` jobs that dispatch `nqa/worker.py` on executors).

