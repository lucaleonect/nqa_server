import json
import os
import time
import psycopg2
from psycopg2.extras import RealDictCursor
import requests


DB_URL = os.environ["DATABASE_URL"]
SOLVER_URL = os.environ.get("SOLVER_URL", "http://solver:8081")
SOLVER_TIMEOUT = os.environ.get("SOLVER_TIMEOUT")

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

    endpoint = f"{SOLVER_URL.rstrip('/')}/run"
    print(f"[scheduler] submitting job {job_id} to solver: {endpoint}")
    try:
        response = requests.post(
            endpoint,
            json={"job_id": job_id},
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
