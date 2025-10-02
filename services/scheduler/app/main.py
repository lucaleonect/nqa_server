import json
import os
import time
from typing import Optional

import numpy as np
import psycopg2
import requests
from psycopg2.extras import RealDictCursor


DB_URL = os.environ["DATABASE_URL"]
SOLVER_URL = os.environ.get("SOLVER_URL", "http://solver:8081")
SOLVER_TIMEOUT = os.environ.get("SOLVER_TIMEOUT")
DATA_ROOT = os.environ.get("DATA_ROOT", "/data")
REQUEST_FILENAME = "study_request.json"

try:
    SOLVER_TIMEOUT = None if SOLVER_TIMEOUT in (None, "") else float(SOLVER_TIMEOUT)
except ValueError:
    print(f"[scheduler] invalid SOLVER_TIMEOUT value '{SOLVER_TIMEOUT}', falling back to no timeout")
    SOLVER_TIMEOUT = None


CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS jobs (
id TEXT PRIMARY KEY,
status TEXT NOT NULL,
filename TEXT NOT NULL,
created_at TIMESTAMP NOT NULL,
updated_at TIMESTAMP NOT NULL,
result_dir TEXT NOT NULL,
error TEXT
);
"""


SELECT_NEXT = """
SELECT * FROM jobs WHERE status = 'QUEUED' ORDER BY created_at ASC LIMIT 1;
"""


MARK_RUNNING = """
UPDATE jobs SET status='RUNNING', updated_at=NOW() WHERE id=%s;
"""


MARK_DONE = """
UPDATE jobs SET status='DONE', updated_at=NOW(), error=NULL WHERE id=%s;
"""


MARK_FAILED = """
UPDATE jobs SET status='FAILED', updated_at=NOW(), error=%s WHERE id=%s;
"""


def _job_dir(job_id: str) -> str:
    return os.path.join(DATA_ROOT, "jobs", job_id)


def _read_study_request(job_id: str) -> dict:
    path = os.path.join(_job_dir(job_id), REQUEST_FILENAME)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid {REQUEST_FILENAME}: {exc}") from exc
    return data if isinstance(data, dict) else {}


def _load_matrix(path: str) -> np.ndarray:
    try:
        matrix = np.load(path, allow_pickle=False)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"missing required matrix: {os.path.basename(path)}") from exc
    except Exception as exc:
        raise ValueError(f"failed to load {path}: {exc}") from exc
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"J.npy must contain a square 2D array, got shape {matrix.shape}")
    return matrix


def _load_vector(path: str) -> Optional[np.ndarray]:
    if not os.path.exists(path):
        return None
    try:
        vector = np.load(path, allow_pickle=False)
    except Exception as exc:
        raise ValueError(f"failed to load {path}: {exc}") from exc
    if vector.ndim != 1:
        raise ValueError(f"{os.path.basename(path)} must be 1D, got shape {vector.shape}")
    return vector


def _int_field(payload: dict, key: str, default: int, minimum: int) -> int:
    value = payload.get(key, default)
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{key} must be an integer")
    if parsed < minimum:
        raise ValueError(f"{key} must be at least {minimum}")
    return parsed


def _build_solver_payload(job: dict) -> dict:
    job_id = job["id"]
    job_directory = _job_dir(job_id)
    if not os.path.isdir(job_directory):
        raise FileNotFoundError(f"job directory missing: {job_directory}")

    study_request = _read_study_request(job_id)

    j_path = os.path.join(job_directory, "J.npy")
    h_path = os.path.join(job_directory, "h_vector.npy")
    g_path = os.path.join(job_directory, "g_vector.npy")

    payload = {
        "J_matrix": _load_matrix(j_path).tolist(),
    }

    study_name = study_request.get("study_name")
    if not study_name:
        result_dir = job.get("result_dir")
        study_name = os.path.basename(result_dir) if result_dir else job_id
    payload["study_name"] = study_name

    study_args = study_request.get("study_args")
    if isinstance(study_args, dict) and study_args:
        payload["study_args"] = study_args

    vector = _load_vector(h_path)
    if vector is not None:
        payload["h_vector"] = vector.tolist()
    vector = _load_vector(g_path)
    if vector is not None:
        payload["g_vector"] = vector.tolist()

    if "cuda_device" in study_request and study_request["cuda_device"] is not None:
        try:
            payload["cuda_device"] = int(study_request["cuda_device"])
        except (TypeError, ValueError) as exc:
            raise ValueError("cuda_device must be an integer") from exc

    return payload

def run_once(conn):
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(SELECT_NEXT)
        job = cur.fetchone()
        if not job:
            return False
        job_id = job["id"]
        print(f"[scheduler] picked job {job_id}")
        cur.execute(MARK_RUNNING, (job_id,))
        conn.commit()

    try:
        payload = _build_solver_payload(job)
    except Exception as exc:
        msg = f"payload assembly failed: {exc}"
        with conn.cursor() as cur:
            cur.execute(MARK_FAILED, (msg[:8000], job_id))
            conn.commit()
        print(f"[scheduler] job {job_id} FAILED: {msg}")
        return True

    endpoint = f"{SOLVER_URL.rstrip('/')}/run"
    study_name = payload.get("study_name")
    print(f"[scheduler] submitting job {job_id} (study={study_name}) to solver: {endpoint}")
    try:
        response = requests.post(
            endpoint,
            json=payload,
            timeout=SOLVER_TIMEOUT,
        )
    except requests.RequestException as exc:
        msg = f"solver request failed: {exc}"
        with conn.cursor() as cur:
            cur.execute(MARK_FAILED, (msg[:8000], job_id))
            conn.commit()
        print(f"[scheduler] job {job_id} FAILED: {msg}")
        return True

    def _parse_solver_response() -> str:
        try:
            body = response.json()
        except ValueError:
            return (response.text or response.reason or "solver returned non-json").strip()
        if isinstance(body, dict):
            detail = body.get("detail") or body.get("error")
            if isinstance(detail, dict):
                return json.dumps(detail)[:8000]
            if isinstance(detail, str):
                return detail.strip()[:8000]
            stdout = body.get("stdout")
            stderr = body.get("stderr")
            fragments = [str(fragment).strip() for fragment in (detail, stderr, stdout) if fragment]
            if fragments:
                return " | ".join(fragments)[:8000]
            return json.dumps(body)[:8000]
        return str(body)[:8000]

    if response.ok:
        print(f"[scheduler] job {job_id} DONE: {response.text.strip()[:500]}")
        with conn.cursor() as cur:
            cur.execute(MARK_DONE, (job_id,))
            conn.commit()
    else:
        msg = _parse_solver_response()
        with conn.cursor() as cur:
            cur.execute(MARK_FAILED, (msg, job_id))
            conn.commit()
        print(f"[scheduler] job {job_id} FAILED: {msg}")
    return True


if __name__ == "__main__":
    while True:
        try:
            with psycopg2.connect(DB_URL) as conn:
                with conn.cursor() as cur:
                    cur.execute(CREATE_TABLE_SQL)
                    conn.commit()
                # Keep using this open connection inside the loop
                while True:
                    did = run_once(conn)
                    time.sleep(2 if did else 4)
        except Exception as e:
            print("[scheduler] DB not ready or error:", e)
            time.sleep(4)
