import os
import subprocess
from pyspark.sql import SparkSession


JOB_ID = os.environ["NQA_JOB_ID"]
DATA_ROOT = os.environ.get("DATA_ROOT", "/data")
JOB_DIR = os.path.join(DATA_ROOT, "jobs", JOB_ID)
RES_DIR = os.path.join(DATA_ROOT, "results", JOB_ID)
J_PATH = os.path.join(JOB_DIR, "J.npy")

def _run_solver(_):
    os.makedirs(RES_DIR, exist_ok=True)
    cmd = ["python3", "/solver/worker.py", "--J_matrix_path", J_PATH]
    print("[executor] running:", " ".join(cmd), "cwd=", RES_DIR, flush=True)
    rc = subprocess.call(cmd, cwd=RES_DIR)
    if rc != 0:
        raise RuntimeError(f"solver exit code {rc}")
    
if __name__ == "__main__":
    spark = SparkSession.builder.appName(f"nqa_job_{JOB_ID}").getOrCreate()
    # one partition → exactly one task → reserves one GPU
    spark.sparkContext.parallelize([1], 1).foreach(_run_solver)
    spark.stop()