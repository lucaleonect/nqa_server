# Neural Quantum Annealing Server

A self-contained platform that runs Neural Quantum Annealing (NQA) experiments behind a simple web dashboard. You drop in Ising or QUBO matrices, the system schedules a run on a GPU-enabled solver, and you can follow along from your browser.

Note: it is also possible to download only the solver code in services/solver and run it directly in a Python environment with GPU access. 

## Repository Tour
- `docker-compose.yml` – wiring between services, shared volumes, and environment variables.
- `.env` – default credentials and Optuna parameters surfaced in the upload form.
- `services/api` – FastAPI UI + REST API implementation.
- `services/scheduler` – background polling worker.
- `services/solver` – solver service, JAX/Optax code (`nqa/`), and container definition. See `services/solver/README.md` for solver-specific notes.
- `shared-data` (Docker volume) – persistent storage for job inputs and study outputs.
- `docs/` – deeper service documentation (`api.md`, `scheduler.md`, `solver.md`).
- `paper/` – reference material and research utilities.

## API Quick Reference
- `GET /` – upload and monitoring UI.
- `POST /upload` – submit an Ising (`J.npy`) or QUBO (`qubo_matrix.npy`) job, plus optional metadata and study args.
- `GET /jobs` – list all jobs.
- `GET /jobs/{job_id}` – get one job's status.
- `GET /jobs/{job_id}/download` – download study artifacts as zip (for `RUNNING`/`DONE`).
- `GET /jobs/{job_id}/dashboard` – return the Optuna dashboard URL for that job.
- `GET /jobs/{job_id}/optuna-db` – download `optuna_db.db` for that job.

## Acknowledgements
- Built on the [Optuna](https://optuna.org/) optimisation framework and the [Optuna Dashboard](https://github.com/optuna/optuna-dashboard).
- CUDA-enabled base image courtesy of NVIDIA’s JAX container.
