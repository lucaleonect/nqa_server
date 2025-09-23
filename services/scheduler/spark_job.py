import json
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
H_path = os.path.join(job_dir, "h_vector.npy")
G_path = os.path.join(job_dir, "g_vector.npy")
args_path = os.path.join(job_dir, "cli_args.json")


os.makedirs(res_dir, exist_ok=True)


# Initialize Spark (even if we don't use executors yet)
spark = SparkSession.builder.appName(f"nqa_job_{job_id}").getOrCreate()


# Run your solver in the results directory so it writes outputs there
solver = "/app/solver/master.py"  # adjust if your solver lives elsewhere
cmd = ["python3", solver, "--J_matrix_path", J_path]
if os.path.exists(H_path):
    cmd.extend(["--h_vector_path", H_path])
if os.path.exists(G_path):
    cmd.extend(["--g_vector_path", G_path])
if os.path.exists(args_path):
    try:
        with open(args_path, "r", encoding="utf-8") as f:
            extra_args = json.load(f)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"failed to load CLI args: {exc}") from exc
    for key, value in extra_args.items():
        if isinstance(value, bool):
            cmd.append(f"--{key}" if value else f"--no-{key}")
        else:
            cmd.extend([f"--{key}", str(value)])


print("[spark_job] running:", " ".join(cmd), " in ", res_dir)
rc = subprocess.call(cmd, cwd=res_dir)


spark.stop()


if rc != 0:
    raise SystemExit(rc)
print("[spark_job] done")
