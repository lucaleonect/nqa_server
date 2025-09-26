import json
import os
import shutil
import tempfile
import uuid
from typing import Dict, Optional, Sequence, cast

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from zipfile import ZipFile

from .db import Base, SessionLocal, engine
from .models import Job, JobStatus


DATA_ROOT = os.environ.get("DATA_ROOT", "/data")
UPLOAD_ROOT = os.path.join(DATA_ROOT, "jobs")
STUDY_ROOT = os.path.join(DATA_ROOT, "studies")
REQUEST_FILENAME = "study_request.json"
os.makedirs(UPLOAD_ROOT, exist_ok=True)
os.makedirs(STUDY_ROOT, exist_ok=True)


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


def _read_study_request(job_id: str) -> Dict[str, object]:
    path = os.path.join(UPLOAD_ROOT, job_id, REQUEST_FILENAME)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


@app.get("/", response_class=HTMLResponse)
async def index():
    with open(os.path.join(os.path.dirname(__file__), "templates", "index.html"), "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


def _coerce_upload(value: Optional[UploadFile | Sequence[UploadFile]]) -> Optional[UploadFile]:
    """Return the first actual upload object or None when the field was absent."""

    if value is None:
        return None

    if isinstance(value, Sequence) and not isinstance(value, (UploadFile, bytes, str)):
        for item in value:
            coerced = _coerce_upload(item)  # type: ignore[arg-type]
            if coerced is not None:
                return coerced
        return None

    if isinstance(value, UploadFile):
        return value

    if hasattr(value, "filename") and hasattr(value, "file"):
        return cast(UploadFile, value)

    return None


@app.post("/upload")
async def upload(
    file: UploadFile = File(...),
    h_vector: Optional[UploadFile | Sequence[UploadFile]] = File(None),
    g_vector: Optional[UploadFile | Sequence[UploadFile]] = File(None),
):
    # Normalise optional inputs that may arrive as lists or other sentinel values
    h_vector = _coerce_upload(h_vector)
    g_vector = _coerce_upload(g_vector)
    if not file.filename.endswith(".npy"):
        return JSONResponse({"error": "Only .npy files accepted"}, status_code=400)

    for optional_file, field_name in ((h_vector, "h_vector"), (g_vector, "g_vector")):
        if optional_file is None:
            continue
        if not optional_file.filename.endswith(".npy"):
            return JSONResponse({"error": f"{field_name} must be a .npy file"}, status_code=400)

    job_id = str(uuid.uuid4())
    job_dir = os.path.join(UPLOAD_ROOT, job_id)
    os.makedirs(job_dir, exist_ok=True)

    # All studies default to the job identifier and fixed Optuna settings.
    study_name = job_id
    study_dir = os.path.join(STUDY_ROOT, study_name)
    os.makedirs(study_dir, exist_ok=True)

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

    study_payload: Dict[str, object] = {
        "study_name": study_name,
    }

    try:
        with open(os.path.join(job_dir, REQUEST_FILENAME), "w", encoding="utf-8") as f:
            json.dump(study_payload, f)
    except Exception as e:
        return JSONResponse({"error": f"failed to save study request: {e}"}, status_code=500)

    job = Job(id=job_id, filename=file.filename, status=JobStatus.QUEUED, result_dir=study_dir)
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
                "study_name": _read_study_request(j.id).get("study_name"),
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
        request_meta = _read_study_request(job_id)
        return {
            "id": j.id,
            "status": j.status.value,
            "filename": j.filename,
            "study_name": request_meta.get("study_name"),
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
