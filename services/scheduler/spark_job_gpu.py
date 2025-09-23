import json
import os
import subprocess
import sys
from pyspark.sql import SparkSession


JOB_ID = os.environ["NQA_JOB_ID"]
DATA_ROOT = os.environ.get("DATA_ROOT", "/data")
JOB_DIR = os.path.join(DATA_ROOT, "jobs", JOB_ID)
RES_DIR = os.path.join(DATA_ROOT, "results", JOB_ID)
J_PATH = os.path.join(JOB_DIR, "J.npy")
H_PATH = os.path.join(JOB_DIR, "h_vector.npy")
G_PATH = os.path.join(JOB_DIR, "g_vector.npy")
ARGS_PATH = os.path.join(JOB_DIR, "cli_args.json")


def _run_solver(_):
    os.makedirs(RES_DIR, exist_ok=True)
    python_bin = os.environ.get("PYSPARK_PYTHON") or sys.executable
    cmd = [python_bin, "/solver/worker.py", "--J_matrix_path", J_PATH]
    if os.path.exists(H_PATH):
        cmd.extend(["--h_vector_path", H_PATH])
    if os.path.exists(G_PATH):
        cmd.extend(["--g_vector_path", G_PATH])
    if os.path.exists(ARGS_PATH):
        try:
            with open(ARGS_PATH, "r", encoding="utf-8") as f:
                extra_args = json.load(f)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"failed to load CLI args: {exc}") from exc
        for key, value in extra_args.items():
            if isinstance(value, bool):
                cmd.append(f"--{key}" if value else f"--no-{key}")
            else:
                cmd.extend([f"--{key}", str(value)])
    print("[executor] running:", " ".join(cmd), "cwd=", RES_DIR, flush=True)
    rc = subprocess.call(cmd, cwd=RES_DIR)
    if rc != 0:
        raise RuntimeError(f"solver exit code {rc}")


if __name__ == "__main__":
    spark = (
        SparkSession.builder.appName(f"nqa_job_{JOB_ID}")
        .config("spark.executor.resource.gpu.amount", "1")
        .config("spark.task.resource.gpu.amount", "1")
        .config("spark.executor.cores", "1")
        .getOrCreate()
    )
    # one partition → exactly one task → reserves one GPU
    spark.sparkContext.parallelize([1], 1).foreach(_run_solver)
    spark.stop()
