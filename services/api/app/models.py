import enum
import uuid
from datetime import datetime
from sqlalchemy import Column, String, DateTime, Enum, Text
from .db import Base


class JobStatus(str, enum.Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    DONE = "DONE"
    FAILED = "FAILED"


class Job(Base):
    __tablename__ = "jobs"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    status = Column(Enum(JobStatus), nullable=False, default=JobStatus.QUEUED)
    filename = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    result_dir = Column(String, nullable=False)
    error = Column(Text, nullable=True)
