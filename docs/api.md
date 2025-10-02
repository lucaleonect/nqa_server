# API Service (`services/api`)

The API service exposes the public-facing interface for submitting jobs, inspecting their status, and downloading solver artefacts.  It is implemented with FastAPI, persists job metadata to PostgreSQL via SQLAlchemy, and writes uploaded files to the shared volume mounted at `/data`.


## Responsibilities

- Validate and store user uploads (`J.npy`, optional `h_vector.npy`, `g_vector.npy`).
- Accept optional user-supplied study names, ensure uniqueness, and record them for downstream services.
- Persist solver metadata (`study_name`) to `study_request.json` alongside the matrix data and seed an empty study directory for downstream services.
- Maintain job records (`QUEUED` → `RUNNING` → `DONE`/`FAILED`) in the `jobs` table.
- Serve a lightweight HTML dashboard (`/`) for manual interactions.
- Package solver outputs into a ZIP archive on demand.
- Provide direct download access to the live Optuna SQLite database for each study.
- Expose Optuna's interactive dashboard and ensure only one instance runs at a time.


## Container Image

- **Dockerfile**: `services/api/Dockerfile`
- **Base**: `python:3.11-slim`
- **Runtime command**: `uvicorn app.main:app --host 0.0.0.0 --port 8000`
- **Ports**: `8000/tcp`
- **Dependencies installed**: `fastapi`, `uvicorn[standard]`, `SQLAlchemy`, `psycopg2-binary`, `python-multipart`, `jinja2`, `optuna-dashboard`


## Environment Variables

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `DATABASE_URL` | ✔ | _none_ | SQLAlchemy connection string to PostgreSQL (`postgresql+psycopg2://user:pass@host:5432/db`). |
| `DATA_ROOT` | ✖ | `/data` | Mount point of the shared volume containing `jobs/` and `studies/`. |


## Directory Usage

- Input matrices are stored under `<DATA_ROOT>/jobs/<job_id>/`.
- Optuna study outputs land under `<DATA_ROOT>/studies/<study_name>/`. The API pre-creates the directory referenced by each job so downstream services can write into it safely.
- The service creates directories on upload if they do not exist.


## REST & UI Endpoints

| Method & Path | Description | Request | Response |
|---------------|-------------|---------|----------|
| `GET /` | HTML upload form rendered from `templates/index.html`. | – | HTML page with fields for matrix uploads; studies run with built-in Optuna defaults. |
| `POST /upload` | Accepts a multipart upload containing the problem matrices. | Fields: `file` (required `.npy` for `J`), `h_vector`, `g_vector` (optional `.npy`), `study_name` (optional unique string), `study_args` (optional JSON object mirroring the solver CLI flags, e.g. `{ "num_trials": 100, "mcmc_num_samples_min": 8 }`). The bundled HTML form exposes dedicated inputs for each of these values, pre-populated with the defaults defined in `nqa/master.py`. | `200 OK` with `{ "job_id": <uuid>, "status": "QUEUED" }` on success. Errors return `400/500` JSON with an `error` key. Conflicting names raise `409`. |
| `GET /jobs` | List jobs ordered by `created_at DESC`. | – | JSON array with `id`, `status`, `filename`, `study_name`, `study_display_name`, `created_at`, `updated_at`, `error`. Timestamps are ISO strings with `Z` suffix. |
| `GET /jobs/{job_id}` | Retrieve one job. | – | Same fields as the list entry, or `404` JSON `{ "error": "not found" }`. |
| `GET /jobs/{job_id}/download` | Package solver results for a completed job. | – | When the job is `DONE` (or still `RUNNING` but producing files), returns a ZIP archive built on the fly from `<result_dir>`. Otherwise `400` with reason. |
| `GET /jobs/{job_id}/optuna-db` | Convenience wrapper to fetch the Optuna database for the job's study. | – | Resolves the underlying study and returns the SQLite file or mirrors the errors from the study endpoint. |
| `GET /jobs/{job_id}/dashboard` | Resolve the public Optuna Dashboard URL for the job's study. | – | Returns `{ "dashboard_url": <url> }` when the dashboard can be started. Automatically shuts down any previously running dashboard instance before serving a new one. Provides `4xx/5xx` JSON errors on failure. |
| `GET /studies/{study_name}/optuna-db` | Fetch the Optuna database (`optuna_db.db`) for a study. | – | Returns the SQLite file even if the study is still running. Responds with `404` if the file is absent and `400` for invalid study names. |


### Study Request Metadata

- Metadata is normalised and written to `<DATA_ROOT>/jobs/<job_id>/study_request.json`.
- `study_name` values default to the job ID to guarantee uniqueness. When users supply `study_name` in the form payload, it is sanitised, deduplicated against existing studies, and stored alongside `requested_study_name` (the original value for UI display).
- Additional solver hyperparameters can be overridden by including a `study_args` JSON object during upload. Allowed keys mirror the CLI flags exposed by `nqa/master.py` (for example, `num_trials`, `num_workers`, the `vqa_*` bounds, `sgd_*` bounds, `sr_diagonal_shift_*`, `dbqs_*`, and `mcmc_*`). Invalid keys or value types are rejected during upload. The web UI renders a textbox for every supported parameter so users can inspect and tweak the solver defaults without crafting JSON by hand.


## Database Schema

`services/api/app/models.py` defines the ORM model for the `jobs` table:

| Column | Type | Description |
|--------|------|-------------|
| `id` | `VARCHAR` (PK) | UUID generated on upload. |
| `status` | `ENUM('QUEUED','RUNNING','DONE','FAILED')` | Updated by API (initial) and scheduler. |
| `filename` | `VARCHAR` | Original filename of the upload. |
| `created_at` | `TIMESTAMP` | Defaults to `datetime.utcnow`. |
| `updated_at` | `TIMESTAMP` | Updated to `NOW()` by API on creation, scheduler on status changes. |
| `result_dir` | `VARCHAR` | Absolute path inside `DATA_ROOT/studies/<study_name>`. |
| `error` | `TEXT` | Optional human-readable error message set by the scheduler. |

The database engine is created with `pool_pre_ping=True` to gracefully handle idle connections.


## Interactions with Other Services

- Writes to the shared volume are consumed by the solver (`study_request.json`, `J.npy`, optional vectors).
- The scheduler relies on the `jobs` table populated here; it never creates jobs itself.
- Study directories are seeded by the API so that the solver can immediately write files.
- When users open the Optuna dashboard (either via the HTML UI link or the `/jobs/{id}/dashboard` endpoint), the API launches the dashboard in a dedicated subprocess and terminates any previous dashboard process to avoid port conflicts.


## Local Development & Testing

1. Install dependencies: `pip install fastapi uvicorn[standard] SQLAlchemy psycopg2-binary python-multipart jinja2 optuna-dashboard`.
2. Set environment variables, e.g.:
   ```bash
   export DATABASE_URL=postgresql+psycopg2://nqa:nqa_password@localhost:5432/nqa
   export DATA_ROOT=/tmp/nqa-data
   ```
3. Start the service: `uvicorn app.main:app --reload --port 8000`.
4. Use the HTML form or call the endpoints via `curl`/`httpie` to exercise the upload pipeline.

There are no automated tests dedicated to the API in this repository yet.  Consider adding FastAPI integration tests that spin up a temporary database (e.g., with `pytest` + `psycopg2` + transaction rollbacks) if you plan to evolve the service.


## Operational Notes

- The upload endpoint rejects files that do not end in `.npy`; this guards against accidental CSV uploads.
- Large files are streamed to disk using `shutil.copyfileobj`; the API does not keep them in memory.
- ZIP downloads create a temporary file using `NamedTemporaryFile(delete=False)`; FastAPI handles cleanup when the response is closed.
- The HTML job list contains an “open dashboard” action for `RUNNING`/`DONE` jobs; clicking it opens the Optuna dashboard in a new tab using the public URL returned by the API.


## Credits

This service embeds the [Optuna](https://optuna.org/) optimization framework and bundles the [Optuna Dashboard](https://github.com/optuna/optuna-dashboard) for live study introspection. We are grateful to both projects and their communities for making these capabilities available under permissive open-source licenses.
