import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from uuid import uuid4

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator, model_validator


app = FastAPI(title="nqa-solver")


DATA_ROOT = Path(os.environ.get("DATA_ROOT", "/data")).resolve()
DATA_ROOT.mkdir(parents=True, exist_ok=True)
MASTER_SCRIPT = Path(os.environ.get("NQA_MASTER_SCRIPT", "/nqa/nqa/master.py")).resolve()
PYTHON_BIN = os.environ.get("NQA_PYTHON_BIN") or sys.executable
DEFAULT_CUDA_DEVICE = os.environ.get("NQA_DEFAULT_CUDA_DEVICE")
MASTER_ROOT = MASTER_SCRIPT.parent
MASTER_STUDIES_ROOT = (MASTER_ROOT / "studies").resolve()


_ALLOWED_STUDY_ARGS: Dict[str, Tuple[str, Optional[float]]] = {
    "num_trials": ("int", 1),
    "num_workers": ("int", 1),
    "trial_max_runtime": ("int", 1),
    "vqa_num_annealing_steps_min": ("int", 1),
    "vqa_num_annealing_steps_max": ("int", 1),
    "vqa_num_updates_per_step_min": ("int", 1),
    "vqa_num_updates_per_step_max": ("int", 1),
    "vqa_annealing_field_scale_min": ("float", 0.0),
    "vqa_annealing_field_scale_max": ("float", 0.0),
    "vqa_catalyst_field_scale_min": ("float", 0.0),
    "vqa_catalyst_field_scale_max": ("float", 0.0),
    "sgd_learning_rate_min": ("float", 0.0),
    "sgd_learning_rate_max": ("float", 0.0),
    "sgd_momentum_min": ("float", 0.0),
    "sgd_momentum_max": ("float", 0.0),
    "sr_diagonal_shift_min": ("float", 0.0),
    "sr_diagonal_shift_max": ("float", 0.0),
    "dbqs_num_hidden_layers": ("int", 1),
    "dbqs_unit_density_per_layer_min": ("float", 0.0),
    "dbqs_unit_density_per_layer_max": ("float", 0.0),
    "mcmc_num_samples_min": ("int", 1),
    "mcmc_num_samples_max": ("int", 1),
    "mcmc_num_sweep_steps_min": ("int", 1),
    "mcmc_num_sweep_steps_max": ("int", 1),
}


class StudyRequest(BaseModel):
    J_matrix: List[List[float]] = Field(..., description="Square coupling matrix")
    h_vector: Optional[List[float]] = Field(None, description="Optional longitudinal field")
    g_vector: Optional[List[float]] = Field(None, description="Optional transverse field")
    study_name: Optional[str] = Field(None, description="Custom identifier for the Optuna study")
    cuda_device: Optional[int] = Field(None, description="Override CUDA device index")
    energy_shift: float = Field(0.0, description="Constant energy offset added to the objective")
    study_args: Dict[str, Union[int, float]] = Field(default_factory=dict, description="Additional study arguments")

    @field_validator("study_args", mode="before")
    @classmethod
    def _sanitize_study_args(cls, value):
        if value in (None, {}):
            return {}
        if not isinstance(value, dict):
            raise ValueError("study_args must be a JSON object mapping argument names to values")

        parsed: Dict[str, Union[int, float]] = {}
        for key, raw in value.items():
            if key not in _ALLOWED_STUDY_ARGS:
                raise ValueError(f"unsupported study argument: {key}")
            expected_type, min_value = _ALLOWED_STUDY_ARGS[key]
            if raw is None:
                continue
            if isinstance(raw, bool):
                raise ValueError(f"{key} must be a number")
            if expected_type == "int":
                if isinstance(raw, float):
                    if not raw.is_integer():
                        raise ValueError(f"{key} must be an integer")
                    converted = int(raw)
                else:
                    try:
                        converted = int(raw)
                    except (TypeError, ValueError) as exc:
                        raise ValueError(f"{key} must be an integer") from exc
            else:
                try:
                    converted = float(raw)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{key} must be a number") from exc
            if min_value is not None and converted < min_value:
                raise ValueError(f"{key} must be at least {min_value}")
            parsed[key] = converted
        return parsed

    @model_validator(mode="after")
    def _vectors_need_square_matrix(cls, model):
        if not model.J_matrix:
            raise ValueError("J_matrix must not be empty")
        return model


def _truncate(text: str, limit: int = 4000) -> str:
    if not text:
        return ""
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[-limit:]


def _ensure_square_matrix(payload: List[List[float]]) -> np.ndarray:
    array = np.asarray(payload, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise HTTPException(status_code=400, detail="J_matrix must be a square 2D array")
    return array


def _ensure_vector(payload: Optional[List[float]], size: int, name: str) -> Optional[np.ndarray]:
    if payload is None:
        return None
    array = np.asarray(payload, dtype=np.float64)
    if array.ndim != 1 or array.shape[0] != size:
        raise HTTPException(status_code=400, detail=f"{name} must be a 1D array of length {size}")
    return array


def _sanitize_name(raw: Optional[str]) -> str:
    if raw:
        cleaned = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in raw.strip())
        cleaned = cleaned.strip("._-")
        if cleaned:
            return cleaned
    return f"study_{uuid4().hex}"


def _prepare_study_dir(study_id: str) -> Path:
    study_dir = (DATA_ROOT / "studies" / study_id).resolve()
    if DATA_ROOT not in study_dir.parents and study_dir != DATA_ROOT:
        raise HTTPException(status_code=400, detail="study path escapes data root")
    study_dir.mkdir(parents=True, exist_ok=True)
    return study_dir


def _persist_inputs(study_dir: Path, j_matrix: np.ndarray, h_vector: Optional[np.ndarray], g_vector: Optional[np.ndarray]):
    inputs_dir = study_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    j_path = inputs_dir / "J.npy"
    np.save(j_path, j_matrix)
    h_path = None
    g_path = None
    if h_vector is not None:
        h_path = inputs_dir / "h_vector.npy"
        np.save(h_path, h_vector)
    if g_vector is not None:
        g_path = inputs_dir / "g_vector.npy"
        np.save(g_path, g_vector)
    return j_path, h_path, g_path


def _build_command(
    study_dir: Path,
    j_path: Path,
    h_path: Optional[Path],
    g_path: Optional[Path],
    request: StudyRequest,
) -> List[str]:
    if not MASTER_SCRIPT.exists():
        raise HTTPException(status_code=500, detail="master script not found inside solver image")

    study_suffix = os.path.relpath(study_dir, MASTER_STUDIES_ROOT)
    cmd: List[str] = [
        PYTHON_BIN,
        str(MASTER_SCRIPT),
        "--J_matrix_path",
        str(j_path),
        "--study_name",
        study_suffix,
    ]

    if h_path is not None:
        cmd.extend(["--h_vector_path", str(h_path)])
    if g_path is not None:
        cmd.extend(["--g_vector_path", str(g_path)])

    if request.energy_shift is not None:
        cmd.extend(["--energy_shift", str(request.energy_shift)])

    cuda_device = request.cuda_device if request.cuda_device is not None else DEFAULT_CUDA_DEVICE
    if cuda_device is not None:
        cmd.extend(["--cuda_device", str(cuda_device)])

    if request.study_args:
        for key in sorted(request.study_args):
            value = request.study_args[key]
            if value is None:
                continue
            cmd.extend([f"--{key}", str(value)])

    return cmd


@app.post("/run")
def run_study(request: StudyRequest):
    j_matrix = _ensure_square_matrix(request.J_matrix)
    size = j_matrix.shape[0]
    h_vector = _ensure_vector(request.h_vector, size, "h_vector")
    g_vector = _ensure_vector(request.g_vector, size, "g_vector")

    study_id = _sanitize_name(request.study_name)
    study_dir = _prepare_study_dir(study_id)
    j_path, h_path, g_path = _persist_inputs(study_dir, j_matrix, h_vector, g_vector)
    cmd = _build_command(study_dir, j_path, h_path, g_path, request)

    env = os.environ.copy()
    env.setdefault("DATA_ROOT", str(DATA_ROOT))
    cuda_device = request.cuda_device if request.cuda_device is not None else DEFAULT_CUDA_DEVICE
    if cuda_device is not None:
        env.setdefault("CUDA_VISIBLE_DEVICES", str(cuda_device))

    print(f"[solver] running study {study_id}: {' '.join(cmd)}", flush=True)
    try:
        result = subprocess.run(
            cmd,
            cwd=str(MASTER_ROOT),
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

    return {
        "status": "ok",
        "stdout": stdout_tail,
        "stderr": stderr_tail,
        "study_id": study_id,
        "study_path": str(study_dir),
    }


@app.get("/health")
def health():
    return {"status": "ready"}
