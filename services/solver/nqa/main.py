import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


app = FastAPI(title="nqa-solver")


DATA_ROOT = Path(os.environ.get("DATA_ROOT", "/data")).resolve()
WORKER_SCRIPT = Path(os.environ.get("NQA_WORKER_SCRIPT", "/nqa/nqa/worker.py")).resolve()
PYTHON_BIN = os.environ.get("NQA_PYTHON_BIN") or sys.executable
DEFAULT_CUDA_DEVICE = os.environ.get("NQA_DEFAULT_CUDA_DEVICE")


class JobRequest(BaseModel):
    job_id: str


def _truncate(text: str, limit: int = 4000) -> str:
    if not text:
        return ""
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[-limit:]


def _validate_paths(job_id: str) -> Tuple[Path, Path, Path]:
    job_dir = (DATA_ROOT / "jobs" / job_id).resolve()
    result_dir = (DATA_ROOT / "results" / job_id).resolve()
    j_matrix_path = job_dir / "J.npy"

    if DATA_ROOT not in job_dir.parents:
        raise HTTPException(status_code=400, detail="job dir outside data root")
    if not j_matrix_path.exists():
        raise HTTPException(status_code=404, detail="J.npy not found for job")
    result_dir.mkdir(parents=True, exist_ok=True)
    return job_dir, result_dir, j_matrix_path


def _load_cli_args(args_path: Path) -> Dict[str, object]:
    try:
        with args_path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"invalid cli args json: {exc}") from exc
    except FileNotFoundError:
        return {}
    if not isinstance(raw, dict):
        raise HTTPException(status_code=400, detail="cli args file must contain an object")
    return raw


def _build_command(job_id: str) -> Tuple[List[str], Path]:
    job_dir, result_dir, j_matrix_path = _validate_paths(job_id)
    if not WORKER_SCRIPT.exists():
        raise HTTPException(status_code=500, detail="worker script not found inside solver image")

    h_path = job_dir / "h_vector.npy"
    g_path = job_dir / "g_vector.npy"
    args_path = job_dir / "cli_args.json"
    cli_args = _load_cli_args(args_path)

    cmd: List[str] = [PYTHON_BIN, str(WORKER_SCRIPT), "--J_matrix_path", str(j_matrix_path)]
    if h_path.exists():
        cmd.extend(["--h_vector_path", str(h_path)])
    if g_path.exists():
        cmd.extend(["--g_vector_path", str(g_path)])

    has_save_path = False
    has_cuda_device = False
    for key, value in sorted(cli_args.items()):
        if value is None:
            continue
        flag = f"--{key}"
        if isinstance(value, bool):
            cmd.append(flag if value else f"--no-{key}")
            if key == "cuda_device" and value:
                has_cuda_device = True
            continue
        cmd.extend([flag, str(value)])
        if key == "save_path":
            has_save_path = True
        if key == "cuda_device":
            has_cuda_device = True

    if not has_save_path:
        cmd.extend(["--save_path", str(result_dir)])
    if not has_cuda_device and DEFAULT_CUDA_DEVICE:
        cmd.extend(["--cuda_device", DEFAULT_CUDA_DEVICE])

    return cmd, result_dir


@app.post("/run")
def run_job(request: JobRequest):
    cmd, result_dir = _build_command(request.job_id)
    env = os.environ.copy()
    env.setdefault("DATA_ROOT", str(DATA_ROOT))
    if DEFAULT_CUDA_DEVICE:
        env.setdefault("CUDA_VISIBLE_DEVICES", DEFAULT_CUDA_DEVICE)

    print(f"[solver] running job {request.job_id}: {' '.join(cmd)}", flush=True)
    try:
        result = subprocess.run(
            cmd,
            cwd=result_dir,
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=f"execution failed: {exc}") from exc

    stdout_tail = _truncate(result.stdout)
    stderr_tail = _truncate(result.stderr)

    if result.returncode != 0:
        raise HTTPException(
            status_code=500,
            detail={
                "error": f"solver exited with code {result.returncode}",
                "stdout": stdout_tail,
                "stderr": stderr_tail,
            },
        )

    return {"status": "ok", "stdout": stdout_tail, "stderr": stderr_tail}


@app.get("/health")
def health():
    return {"status": "ready"}

