# Solver Service (`services/solver`)

The solver service hosts the Neural Quantum Annealer (NQA) runtime behind a simple HTTP facade.  It receives job IDs from the scheduler, resolves them to files on the shared volume, and invokes `nqa/worker.py` with the appropriate CLI arguments.  The service is designed to run on a CUDA-capable host and bundles all required Python dependencies inside the container.


## Responsibilities

- Validate job directories and required inputs (`J.npy`).
- Load optional CLI argument overrides from `<DATA_ROOT>/jobs/<job_id>/cli_args.json`.
- Construct a deterministic subprocess command targeting `nqa/worker.py`.
- Ensure result directories exist and serve as the subprocess working directory.
- Capture stdout/stderr (tail truncated to 4 000 chars) and surface them in the HTTP response.
- Translate non-zero exit codes into structured HTTP 500 responses so the scheduler can persist meaningful error messages.


## Container Image

- **Dockerfile**: `services/solver/Dockerfile`
- **Base**: `nvcr.io/nvidia/jax:25.08-py3` (CUDA-enabled JAX runtime)
- **Runtime command**: `uvicorn app.main:app --host 0.0.0.0 --port 8081`
- **Ports**: `8081/tcp`
- **Dependencies installed**: requirements from `services/solver/requirements.txt` (FastAPI, uvicorn, Optuna, pytest utilities, matplotlib, etc.).
- **Application layout inside container**: `/nqa/app` (FastAPI service) and `/nqa/nqa` (JAX solver package).


## API Endpoints

| Method & Path | Description | Response |
|---------------|-------------|----------|
| `GET /health` | Simple readiness endpoint. | `{ "status": "ready" }` |
| `POST /run` | Execute a job. Request body: `{ "job_id": "<uuid>" }`. | `200 OK`: `{ "status": "ok", "stdout": <tail>, "stderr": <tail> }`.  `500`: `{ "detail": { "error": "…", "stdout": "…", "stderr": "…" } }` or a string describing the failure. |

`stdout`/`stderr` tails are appended to aid debugging without transferring entire logs.  Full artefacts remain on disk under `/data/results/<job_id>`.


## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `DATA_ROOT` | `/data` | Root path for the shared volume. Must contain `jobs` and `results` subdirectories. |
| `NQA_WORKER_SCRIPT` | `/nqa/nqa/worker.py` | Path to the solver CLI entrypoint. Useful when testing custom branches or alternate runners. |
| `NQA_PYTHON_BIN` | runtime Python | Interpreter used to launch the worker (e.g., `/opt/conda/bin/python`). |
| `NQA_DEFAULT_CUDA_DEVICE` | unset | If provided, appended to the command via `--cuda_device <value>` and exported as `CUDA_VISIBLE_DEVICES` when the job arguments do not supply one. |

The scheduler’s request does not include any additional metadata.  Solver behaviour is therefore driven entirely by the files present under `<DATA_ROOT>/jobs/<job_id>`.


## Command Construction Details

1. Resolve paths using `pathlib.Path.resolve()`.  The solver guards against directory traversal by verifying that job directories sit underneath `DATA_ROOT`.
2. Required file: `J.npy`. Missing files raise an HTTP 404.
3. Optional files: `h_vector.npy`, `g_vector.npy`, `cli_args.json`.
4. CLI arguments are iterated in sorted order for deterministic command lines.  Booleans become `--flag` or `--no-flag`; other values are stringified.
5. If the user-specified CLI contains `--save_path`, it is honoured. Otherwise `--save_path <DATA_ROOT>/results/<job_id>` is appended to keep outputs co-located.
6. CUDA device handling:
   - If `cli_args.json` already sets `cuda_device`, no changes are made.
   - If `NQA_DEFAULT_CUDA_DEVICE` is set, both `--cuda_device` and `CUDA_VISIBLE_DEVICES` are injected so JAX binds to the intended GPU.
7. Subprocess execution occurs via `subprocess.run` with `cwd` pointing to the results directory and `capture_output=True`.


## Error Semantics

- Exit code `0`: HTTP 200 with truncated logs.
- Non-zero exit code: HTTP 500 with a structured payload containing the exit code and trailing stderr/stdout text.
- Missing worker script or input files: HTTP 500 / 404 respectively.
- JSON decode errors in `cli_args.json`: HTTP 400 with details about the parse failure.

The scheduler records the `detail` field in the database so operators can diagnose failures without logging into the solver container.


## Working with `nqa/`

- `nqa/worker.py` defines the command-line interface consumed by the solver.  Ensure new flags maintain backward compatibility with the API’s form definitions.
- `nqa/master.py` and the `nqa/tests/` suite remain available for standalone experimentation.
- GPU-related utilities (e.g., `nqa/cuda_check.py`) can help verify that devices are enumerated correctly from within the container.


## Local Development & Testing

1. Install requirements: `pip install -r services/solver/requirements.txt` (requires CUDA-compatible environment for full functionality).
2. Start the API: `uvicorn app.main:app --reload --port 8081` from `services/solver`.
3. Set `DATA_ROOT` to a directory containing the expected job structure:
   ```bash
   export DATA_ROOT=$(pwd)/../../shared-data
   mkdir -p "$DATA_ROOT/jobs/$JOB" "$DATA_ROOT/results/$JOB"
   cp J.npy "$DATA_ROOT/jobs/$JOB/J.npy"
   ```
4. Trigger the endpoint with `curl`:
   ```bash
   curl -X POST http://localhost:8081/run \
        -H 'Content-Type: application/json' \
        -d '{"job_id": "'$JOB'"}'
   ```

Run solver unit tests with:

```bash
cd services/solver/nqa
python -m pytest
```

Some tests expect a GPU; for CPU-only environments, configure JAX appropriately (e.g., `export XLA_FLAGS=--xla_force_host_platform_device_count=1`).


## Operational Notes

- The container requests `gpus: all` in `docker-compose.yml`.  Adjust to a specific count or device list when deploying in shared GPU environments.
- Standard output includes a line such as `[solver] running job <id>: python /nqa/nqa/worker.py …` for traceability.
- Subprocess stdout/stderr is flushed after completion; long-running jobs should stream their own progress to files inside the results directory if live feedback is needed.
- Keep an eye on disk usage within the shared volume; results can be large depending on solver configuration.

