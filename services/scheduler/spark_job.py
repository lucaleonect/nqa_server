import os
import argparse
import subprocess
from pyspark.sql import SparkSession


parser = argparse.ArgumentParser()
parser.add_argument("--job-id", required=True)
args = parser.parse_args()


job_id = args.job_id
DATA_ROOT = os.environ.get("DATA_ROOT", "/data")
job_dir = os.path.join(DATA_ROOT, "jobs", job_id)
res_dir = os.path.join(DATA_ROOT, "results", job_id)
J_path = os.path.join(job_dir, "J.npy")


os.makedirs(res_dir, exist_ok=True)


# Initialize Spark (even if we don't use executors yet)
spark = SparkSession.builder.appName(f"nqa_job_{job_id}").getOrCreate()


# Run your solver in the results directory so it writes outputs there
solver = "/app/solver/master.py"  # adjust if your solver lives elsewhere
cmd = ["python3", solver, "--J_matrix_path", J_path]


print("[spark_job] running:", " ".join(cmd), " in ", res_dir)
rc = subprocess.call(cmd, cwd=res_dir)


spark.stop()


if rc != 0:
    raise SystemExit(rc)
print("[spark_job] done")
