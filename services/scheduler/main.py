import os
import time
import subprocess
from datetime import datetime
import psycopg2
from psycopg2.extras import RealDictCursor


DB_URL = os.environ["DATABASE_URL"]
SPARK_MASTER_URL = os.environ.get("SPARK_MASTER_URL", "spark://spark-master:7077")
DATA_ROOT = os.environ.get("DATA_ROOT", "/data")


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


INIT_TIMESTAMPS = """
UPDATE jobs SET created_at=COALESCE(created_at, NOW()), updated_at=COALESCE(updated_at, NOW());
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

    # submit spark job (client mode). spark_job.py will run the solver.
    cmd = [
        "spark-submit",
        "--master",
        SPARK_MASTER_URL,
        # one executor, one core, one GPU
        "--conf",
        "spark.executor.instances=1",
        "--conf",
        "spark.executor.cores=1",
        "--conf",
        "spark.task.resource.gpu.amount=1",
        "--conf",
        "spark.executor.resource.gpu.amount=1",
        # Bubble env down to the executor
        "--conf",
        f"spark.executorEnv.NQA_JOB_ID={job_id}",
        "--conf",
        f"spark.executorEnv.DATA_ROOT={DATA_ROOT}",
        "/app/spark_job_gpu.py",
    ]
    print("[scheduler] spark-submit:", " ".join(cmd))
    try:
        res = subprocess.run(cmd, check=False, capture_output=True, text=True)
        print(res.stdout)
        if res.returncode == 0:
            with conn.cursor() as cur:
                cur.execute(MARK_DONE, (job_id,))
                conn.commit()
            print(f"[scheduler] job {job_id} DONE")
        else:
            msg = (res.stderr or "spark-submit failed").strip()[:8000]
            with conn.cursor() as cur:
                cur.execute(MARK_FAILED, (msg, job_id))
                conn.commit()
            print(f"[scheduler] job {job_id} FAILED: {msg}")
    except Exception as e:
        with conn.cursor() as cur:
            cur.execute(MARK_FAILED, (str(e), job_id))
            conn.commit()
        print(f"[scheduler] job {job_id} EXCEPTION: {e}")
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
