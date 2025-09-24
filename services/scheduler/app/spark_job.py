import json
import os
import subprocess
import sys
from pyspark.sql import SparkSession
from pyspark import TaskContext


JOB_ID = os.environ["NQA_JOB_ID"]
DATA_ROOT = os.environ.get("DATA_ROOT", "/data")
JOB_DIR = os.path.join(DATA_ROOT, "jobs", JOB_ID)
RES_DIR = os.path.join(DATA_ROOT, "results", JOB_ID)
J_PATH = os.path.join(JOB_DIR, "J.npy")
H_PATH = os.path.join(JOB_DIR, "h_vector.npy")
G_PATH = os.path.join(JOB_DIR, "g_vector.npy")
ARGS_PATH = os.path.join(JOB_DIR, "cli_args.json")
WORKER_SCRIPT = os.environ.get("NQA_WORKER_SCRIPT", "/app/solver/nqa/worker.py")


def _run_solver(_):
    os.makedirs(RES_DIR, exist_ok=True)
    python_bin = os.environ.get("PYSPARK_PYTHON") or sys.executable
    worker_path = os.path.abspath(WORKER_SCRIPT)
    if not os.path.exists(worker_path):
        raise RuntimeError(f"solver entrypoint not found: {worker_path}")

    cmd = [python_bin, worker_path, "--J_matrix_path", J_PATH]
    cmd_has_cuda_device = False
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
                if key == "cuda_device" and value:
                    cmd_has_cuda_device = True
            else:
                cmd.extend([f"--{key}", str(value)])
                if key == "cuda_device":
                    cmd_has_cuda_device = True

    assigned_cuda_devices = None
    try:
        context = TaskContext.get()
        if context is not None:
            gpu_resource = context.resources().get("gpu")
            if gpu_resource and gpu_resource.addresses:
                assigned_cuda_devices = ",".join(gpu_resource.addresses)
    except Exception as exc:  # pragma: no cover - defensive logging only
        print(
            f"[executor] warning: unable to inspect spark task GPU allocation: {exc}",
            file=sys.stderr,
            flush=True,
        )

    if assigned_cuda_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = assigned_cuda_devices
        if not cmd_has_cuda_device:
            first_device = assigned_cuda_devices.split(",", 1)[0]
            cmd.extend(["--cuda_device", first_device])
    print("[executor] running:", " ".join(cmd), "cwd=", RES_DIR, flush=True)
    result = subprocess.run(cmd, cwd=RES_DIR, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout, end="", flush=True)
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr, flush=True)
    if result.returncode != 0:
        stderr_excerpt = (result.stderr or "").strip().splitlines()
        detail = stderr_excerpt[-1] if stderr_excerpt else "no stderr captured"
        raise RuntimeError(f"solver failed (exit code {result.returncode}): {detail}")


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
