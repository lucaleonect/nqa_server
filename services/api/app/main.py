import os
import shutil
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from sqlalchemy.orm import Session
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
async def upload(file: UploadFile = File(...)):
    if not file.filename.endswith(".npy"):
        return JSONResponse({"error": "Only .npy files accepted"}, status_code=400)

    job_id = str(uuid.uuid4())
    job_dir = os.path.join(UPLOAD_ROOT, job_id)
    res_dir = os.path.join(RESULT_ROOT, job_id)
    os.makedirs(job_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)

    dest = os.path.join(job_dir, "J.npy")
    try:
        with open(dest, "wb") as out:
            shutil.copyfileobj(file.file, out)
    except Exception as e:
        return JSONResponse({"error": f"failed to save file: {e}"}, status_code=500)

    job = Job(id=job_id, filename=file.filename, status=JobStatus.QUEUED, result_dir=res_dir)
    with SessionLocal() as db:
        db.add(job)
        db.commit()

    return {"job_id": job_id, "status": job.status}


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
