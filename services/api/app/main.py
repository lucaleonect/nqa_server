import json
import os
import shutil
from typing import Any, Dict, Optional

from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from .db import Base, engine, SessionLocal
from .models import Job, JobStatus
from zipfile import ZipFile
import tempfile
import uuid

DATA_ROOT = os.environ.get("DATA_ROOT", "/data")
UPLOAD_ROOT = os.path.join(DATA_ROOT, "jobs")
RESULT_ROOT = os.path.join(DATA_ROOT, "results")
os.makedirs(UPLOAD_ROOT, exist_ok=True)
os.makedirs(RESULT_ROOT, exist_ok=True)


CLI_ARGUMENT_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    "prng_seed": {"type": "int", "default": None},
    "h_value": {"type": "float", "default": None},
    "g_value": {"type": "float", "default": None},
    "energy_shift": {"type": "float", "default": 0.0},
    "vqa_num_annealing_steps": {"type": "int", "default": 10000},
    "vqa_num_warmup_steps": {"type": "int", "default": 1},
    "vqa_num_updates_per_step": {"type": "int", "default": 1},
    "vqa_num_finetuning_steps": {"type": "int", "default": 100},
    "vqa_annealing_field_scale": {"type": "float", "default": 1.0},
    "vqa_catalyst_field_scale": {"type": "float", "default": 1.0},
    "vqa_no_catalyst": {"type": "bool", "default": False},
    "sgd_learning_rate": {"type": "float", "default": 0.1},
    "sgd_momentum": {"type": "float", "default": 0.5},
    "sr_prefactor": {"type": "complex", "default": "1.0+0.0j"},
    "sr_diagonal_shift": {"type": "float", "default": 0.01},
    "dbqs_num_hidden_layers": {"type": "int", "default": 2},
    "dbqs_unit_density_per_layer": {"type": "float", "default": 1.0},
    "dbqs_param_dtype": {"type": "toggle", "default": "complex", "alt": "float"},
    "dbqs_use_bias": {"type": "bool", "default": True},
    "mcmc_num_samples": {"type": "int", "default": 2**7},
    "mcmc_num_chains": {"type": "int", "default": None},
    "mcmc_num_thermalization_steps": {"type": "int", "default": 2**7},
    "mcmc_num_sweep_steps": {"type": "int", "default": 2**4},
    "mcmc_disable_persistent_markov_chains": {"type": "bool", "default": False},
}


app = FastAPI(title="nqa-server")


@app.on_event("startup")
def startup():
    Base.metadata.create_all(bind=engine)


# very small dependency helper
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/", response_class=HTMLResponse)
async def index():
    with open(os.path.join(os.path.dirname(__file__), "templates", "index.html"), "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.post("/upload")
async def upload(
    request: Request,
    file: UploadFile = File(...),
    h_vector: Any = File(None),
    g_vector: Any = File(None),
):
    # Normalize optional files: treat non-UploadFile values (e.g., empty strings) as absent
    h_vector = h_vector if isinstance(h_vector, UploadFile) else None
    g_vector = g_vector if isinstance(g_vector, UploadFile) else None
    if not file.filename.endswith(".npy"):
        return JSONResponse({"error": "Only .npy files accepted"}, status_code=400)

    for optional_file, field_name in ((h_vector, "h_vector"), (g_vector, "g_vector")):
        if optional_file is None:
            continue
        if not optional_file.filename.endswith(".npy"):
            return JSONResponse({"error": f"{field_name} must be a .npy file"}, status_code=400)

    form = await request.form()

    cli_args = {}
    for arg_name, meta in CLI_ARGUMENT_DEFINITIONS.items():
        arg_type = meta["type"]
        if arg_type == "bool":
            values = form.getlist(arg_name)
            if not values:
                continue
            default_value = bool(meta.get("default", False))
            cli_args[arg_name] = not default_value
            continue
        if arg_type == "toggle":
            values = form.getlist(arg_name)
            if not values:
                continue
            alt_value = meta.get("alt")
            if alt_value is None:
                continue
            cli_args[arg_name] = alt_value
            continue

        raw_value = form.get(arg_name)
        if raw_value is None:
            continue
        value_str = str(raw_value).strip()
        if value_str == "":
            continue
        try:
            if arg_type == "int":
                cli_args[arg_name] = int(value_str)
            elif arg_type == "float":
                cli_args[arg_name] = float(value_str)
            elif arg_type == "complex":
                complex(value_str)
                cli_args[arg_name] = value_str
            else:
                cli_args[arg_name] = value_str
        except ValueError:
            return JSONResponse({"error": f"{arg_name} must be a valid {arg_type}"}, status_code=400)

    if cli_args.get("dbqs_param_dtype") == "float" and not cli_args.get("vqa_no_catalyst", False):
        return JSONResponse(
            {"error": "When using float parameter dtype, the catalyst must be disabled."},
            status_code=400,
        )

    if h_vector is not None and "h_value" in cli_args:
        return JSONResponse(
            {"error": "Provide either h_vector file or h_value, not both"},
            status_code=400,
        )
    if g_vector is not None and "g_value" in cli_args:
        return JSONResponse(
            {"error": "Provide either g_vector file or g_value, not both"},
            status_code=400,
        )

    job_id = str(uuid.uuid4())
    job_dir = os.path.join(UPLOAD_ROOT, job_id)
    res_dir = os.path.join(RESULT_ROOT, job_id)
    os.makedirs(job_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)

    dest = os.path.join(job_dir, "J.npy")
    try:
        file.file.seek(0)
        with open(dest, "wb") as out:
            shutil.copyfileobj(file.file, out)
    except Exception as e:
        return JSONResponse({"error": f"failed to save file: {e}"}, status_code=500)

    try:
        if h_vector is not None:
            h_vector.file.seek(0)
            with open(os.path.join(job_dir, "h_vector.npy"), "wb") as out:
                shutil.copyfileobj(h_vector.file, out)
        if g_vector is not None:
            g_vector.file.seek(0)
            with open(os.path.join(job_dir, "g_vector.npy"), "wb") as out:
                shutil.copyfileobj(g_vector.file, out)
    except Exception as e:
        return JSONResponse({"error": f"failed to save vector file: {e}"}, status_code=500)

    if cli_args:
        default_marker = object()
        sample_value = cli_args.get(
            "mcmc_num_samples", CLI_ARGUMENT_DEFINITIONS["mcmc_num_samples"]["default"]
        )
        filtered_args: Dict[str, Any] = {}
        for arg_name, value in cli_args.items():
            meta = CLI_ARGUMENT_DEFINITIONS.get(arg_name, {})
            default_value = meta.get("default", default_marker)
            if default_value is not default_marker and value == default_value:
                continue
            filtered_args[arg_name] = value

        if "mcmc_num_chains" in filtered_args and filtered_args["mcmc_num_chains"] == sample_value:
            del filtered_args["mcmc_num_chains"]

        if filtered_args:
            try:
                with open(os.path.join(job_dir, "cli_args.json"), "w", encoding="utf-8") as f:
                    json.dump({k: filtered_args[k] for k in sorted(filtered_args)}, f)
            except Exception as e:
                return JSONResponse({"error": f"failed to save argument file: {e}"}, status_code=500)

    job = Job(id=job_id, filename=file.filename, status=JobStatus.QUEUED, result_dir=res_dir)
    with SessionLocal() as db:
        db.add(job)
        db.commit()
        status = job.status.value

    return {"job_id": job_id, "status": status}


@app.get("/jobs")
def list_jobs():
    with SessionLocal() as db:
        jobs = db.query(Job).order_by(Job.created_at.desc()).all()
        return [
            {
                "id": j.id,
                "status": j.status.value,
                "filename": j.filename,
                "created_at": j.created_at.isoformat() + "Z",
                "updated_at": j.updated_at.isoformat() + "Z",
                "error": j.error,
            }
            for j in jobs
        ]


@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    with SessionLocal() as db:
        j = db.get(Job, job_id)
        if not j:
            return JSONResponse({"error": "not found"}, status_code=404)
        return {
            "id": j.id,
            "status": j.status.value,
            "filename": j.filename,
            "created_at": j.created_at.isoformat() + "Z",
            "updated_at": j.updated_at.isoformat() + "Z",
            "error": j.error,
        }


@app.get("/jobs/{job_id}/download")
def download_results(job_id: str):
    with SessionLocal() as db:
        j = db.get(Job, job_id)
        if not j:
            return JSONResponse({"error": "not found"}, status_code=404)
        if j.status != JobStatus.DONE:
            return JSONResponse({"error": f"job not done (status={j.status.value})"}, status_code=400)
        # zip the result dir safely into a NamedTemporaryFile
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=f"_{job_id}.zip")
        tmp_zip = tmp.name
        tmp.close()
        with ZipFile(tmp_zip, "w") as z:
            for root, _, files in os.walk(j.result_dir):
                for fn in files:
                    fp = os.path.join(root, fn)
                    arc = os.path.relpath(fp, j.result_dir)
                    z.write(fp, arc)
        return FileResponse(tmp_zip, filename=f"results_{job_id}.zip")