"""FastAPI application exposing the Neural Quantum Annealing API service."""

import inspect
import io
import json
import logging
import multiprocessing
import os
import shutil
import tempfile
import threading
import urllib.parse
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Tuple, Union, cast

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from zipfile import ZipFile

import numpy as np

from .db import Base, SessionLocal, engine
from .models import Job, JobStatus
from .qubo import QUBO_to_Ising

try:
    import optuna_dashboard
except ImportError:  # pragma: no cover - optional dependency for dashboard UI
    optuna_dashboard = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency resolved at runtime
    import uvicorn
except ImportError:  # pragma: no cover - optional dependency for dashboard UI
    uvicorn = None  # type: ignore[assignment]

DATA_ROOT = os.environ["DATA_ROOT"]
UPLOAD_ROOT = os.path.join(DATA_ROOT, "jobs")
STUDY_ROOT = os.path.join(DATA_ROOT, "studies")
REQUEST_FILENAME = "study_request.json"
os.makedirs(UPLOAD_ROOT, exist_ok=True)
os.makedirs(STUDY_ROOT, exist_ok=True)


logger = logging.getLogger(__name__)


def _load_env_study_default(
    key: str, spec_kind: str, default: Union[int, float], minimum: Optional[float]
) -> Union[int, float]:
    """Resolve a study argument default from the environment, falling back to code defaults."""
    env_key = f"HPO_DEFAULT_{key.upper()}"
    raw_value = os.environ.get(env_key)
    if raw_value is None or not str(raw_value).strip():
        return default

    try:
        coerced = int(raw_value) if spec_kind == "int" else float(raw_value)
    except (TypeError, ValueError):
        logging.getLogger(__name__).warning(
            "Invalid value for %s=%r; falling back to %s", env_key, raw_value, default
        )
        return default

    if minimum is not None and coerced < minimum:
        logging.getLogger(__name__).warning(
            "Value for %s below minimum %s; falling back to %s", env_key, minimum, default
        )
        return default

    return coerced


def _build_spec(
    key: str,
    label: str,
    spec_kind: str,
    default: Union[int, float],
    minimum: Optional[float],
) -> Tuple[str, "_StudyArgSpec"]:
    """Create a study argument specification tuple used to render the HTML form."""
    return (
        key,
        _StudyArgSpec(
            label,
            spec_kind,
            _load_env_study_default(key, spec_kind, default, minimum),
            minimum,
        ),
    )


app = FastAPI(title="nqa-server")

templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))


@dataclass(frozen=True)
class _StudyArgSpec:
    label: str
    kind: str
    default: Union[int, float]
    minimum: Optional[float]


_STUDY_ARG_SPECS: "OrderedDict[str, _StudyArgSpec]" = OrderedDict(
    (
        _build_spec("num_trials", "Number of trials", "int", 100, 1),
        _build_spec("num_workers", "Number of parallel workers", "int", 1, 1),
        _build_spec("trial_max_runtime", "Trial max runtime (seconds)", "int", 60 * 30, 1),
        _build_spec("vqa_num_annealing_steps_min", "VQA annealing steps (min)", "int", 1000, 1),
        _build_spec("vqa_num_annealing_steps_max", "VQA annealing steps (max)", "int", 10000, 1),
        _build_spec("vqa_num_updates_per_step_min", "VQA updates per step (min)", "int", 1, 1),
        _build_spec("vqa_num_updates_per_step_max", "VQA updates per step (max)", "int", 5, 1),
        _build_spec("vqa_annealing_field_scale_min", "Annealing field scale (min)", "float", 1e-1, 0.0),
        _build_spec("vqa_annealing_field_scale_max", "Annealing field scale (max)", "float", 1e1, 0.0),
        _build_spec("vqa_catalyst_field_scale_min", "Catalyst field scale (min)", "float", 1e-1, 0.0),
        _build_spec("vqa_catalyst_field_scale_max", "Catalyst field scale (max)", "float", 1e1, 0.0),
        _build_spec("sgd_learning_rate_min", "SGD learning rate (min)", "float", 1e-3, 0.0),
        _build_spec("sgd_learning_rate_max", "SGD learning rate (max)", "float", 1e0, 0.0),
        _build_spec("sgd_momentum_min", "SGD momentum (min)", "float", 0.0, 0.0),
        _build_spec("sgd_momentum_max", "SGD momentum (max)", "float", 0.9, 0.0),
        _build_spec("sr_diagonal_shift_min", "SR diagonal shift (min)", "float", 1e-9, 0.0),
        _build_spec("sr_diagonal_shift_max", "SR diagonal shift (max)", "float", 1e-2, 0.0),
        _build_spec("dbqs_num_hidden_layers", "DBQS hidden layers", "int", 2, 1),
        _build_spec(
            "dbqs_unit_density_per_layer_min",
            "DBQS unit density per layer (min)",
            "float",
            0.25,
            0.0,
        ),
        _build_spec(
            "dbqs_unit_density_per_layer_max",
            "DBQS unit density per layer (max)",
            "float",
            4.0,
            0.0,
        ),
        _build_spec("mcmc_num_samples_min", "MCMC samples (min)", "int", 2**3, 1),
        _build_spec("mcmc_num_samples_max", "MCMC samples (max)", "int", 2**6, 1),
        _build_spec("mcmc_num_sweep_steps_min", "MCMC sweep steps (min)", "int", 2**2, 1),
        _build_spec("mcmc_num_sweep_steps_max", "MCMC sweep steps (max)", "int", 2**6, 1),
    )
)


_ALLOWED_STUDY_ARGS: Dict[str, Tuple[str, Optional[float]]] = {
    key: (spec.kind, spec.minimum)
    for key, spec in _STUDY_ARG_SPECS.items()
}


def _parse_study_args(raw: Optional[str]) -> Dict[str, object]:
    """Parse the JSON payload carrying Optuna study overrides from the upload form."""
    if raw is None:
        return {}

    text = raw.strip()
    if not text:
        return {}

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"study arguments must be valid JSON: {exc.msg}") from exc

    if not isinstance(payload, dict):
        raise ValueError("study arguments must be a JSON object with key/value pairs")

    parsed: Dict[str, object] = {}
    for key, value in payload.items():
        if key not in _ALLOWED_STUDY_ARGS:
            raise ValueError(f"unsupported study argument: {key}")
        expected_type, min_value = _ALLOWED_STUDY_ARGS[key]
        if value is None:
            continue
        if isinstance(value, bool):
            raise ValueError(f"{key} must be a number")
        if expected_type == "int":
            if isinstance(value, float):
                if not value.is_integer():
                    raise ValueError(f"{key} must be an integer")
                converted = int(value)
            else:
                try:
                    converted = int(value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{key} must be an integer") from exc
        else:
            try:
                converted = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{key} must be a number") from exc
        if min_value is not None and converted < min_value:
            raise ValueError(f"{key} must be at least {min_value}")
        parsed[key] = converted

    return parsed


_dashboard_available = optuna_dashboard is not None and hasattr(optuna_dashboard, "run_server")

if not _dashboard_available:
    if optuna_dashboard is None:
        logger.info("optuna-dashboard package not installed; dashboard endpoint disabled")
    else:
        logger.warning("optuna_dashboard.run_server not available; dashboard endpoint disabled")


def _normalize_dashboard_path(path: Optional[str]) -> str:
    if not path:
        return "/"
    path = path.strip()
    if not path:
        return "/"
    if not path.startswith("/"):
        path = "/" + path
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")
    return path


_DASHBOARD_BIND_HOST = os.environ["OPTUNA_DASHBOARD_BIND_HOST"]
_DASHBOARD_PATH = _normalize_dashboard_path(os.environ.get("OPTUNA_DASHBOARD_PATH"))
_DASHBOARD_ALLOW_ORIGIN = os.environ.get("OPTUNA_DASHBOARD_ALLOW_ORIGIN")


@dataclass
class _DashboardServer:
    storage: str
    study_name: Optional[str]
    host: str
    port: int
    stop: Callable[[], None]
    thread: Optional[threading.Thread] = None
    process: Optional[multiprocessing.Process] = None

    def is_alive(self) -> bool:
        if self.thread is not None:
            return self.thread.is_alive()
        if self.process is not None:
            return self.process.is_alive()
        return False


def _dashboard_process_create_app(storage_url: str, kwargs: Dict[str, object], config_kwargs: Dict[str, object]) -> None:
    try:
        import optuna_dashboard  # type: ignore[import-not-found]
        import uvicorn  # type: ignore[import-not-found]
    except ImportError:  # pragma: no cover - child process best effort
        logging.getLogger(__name__).error("optuna-dashboard or uvicorn unavailable in dashboard process")
        return

    create_app = getattr(optuna_dashboard, "create_app", None)
    if create_app is None:
        logging.getLogger(__name__).error("optuna-dashboard create_app not available in dashboard process")
        return

    try:
        app = create_app(**kwargs)
        config = uvicorn.Config(app, **config_kwargs)
        server = uvicorn.Server(config)
        server.run()
    except Exception as exc:  # pragma: no cover - defensive logging only
        logging.getLogger(__name__).exception("optuna-dashboard server stopped for %s: %s", storage_url, exc)


def _dashboard_process_run_server(storage_url: str, args: Sequence[object], kwargs: Dict[str, object]) -> None:
    try:
        import optuna_dashboard  # type: ignore[import-not-found]
    except ImportError:  # pragma: no cover - child process best effort
        logging.getLogger(__name__).error("optuna-dashboard unavailable in dashboard process")
        return

    try:
        optuna_dashboard.run_server(*args, **kwargs)  # type: ignore[misc]
    except Exception as exc:  # pragma: no cover - defensive logging only
        logging.getLogger(__name__).exception("optuna-dashboard server stopped for %s: %s", storage_url, exc)


_DASHBOARD_PORT = int(os.environ["OPTUNA_DASHBOARD_PORT"])

_dashboard_server: Optional[_DashboardServer] = None
_dashboard_lock = threading.Lock()


def _stop_dashboard_server_locked() -> None:
    global _dashboard_server
    if not _dashboard_server:
        return

    server = _dashboard_server
    _dashboard_server = None

    try:
        server.stop()
    except Exception as exc:  # pragma: no cover - defensive logging only
        logger.exception("failed to stop optuna-dashboard server: %s", exc)


def _start_dashboard_server(storage_url: str, study_name: Optional[str], host: str, port: int) -> Optional[_DashboardServer]:
    server = _start_dashboard_server_with_create_app(storage_url, study_name, host, port)
    if server is not None:
        return server

    return _start_dashboard_server_legacy(storage_url, study_name, host, port)


def _start_dashboard_server_with_create_app(
    storage_url: str, study_name: Optional[str], host: str, port: int
) -> Optional[_DashboardServer]:
    if not _dashboard_available or optuna_dashboard is None or uvicorn is None:
        return None

    create_app = getattr(optuna_dashboard, "create_app", None)
    if create_app is None:
        return None

    params = inspect.signature(create_app).parameters
    if "storage" not in params:
        logger.debug("optuna-dashboard create_app missing 'storage' parameter; falling back to legacy runner")
        return None

    kwargs: Dict[str, object] = {"storage": storage_url}

    if study_name:
        if "study_name" in params:
            kwargs["study_name"] = study_name
        elif "study_names" in params:
            kwargs["study_names"] = [study_name]

    if "path" in params:
        kwargs["path"] = _DASHBOARD_PATH

    if _DASHBOARD_ALLOW_ORIGIN and "allow_websocket_origin" in params:
        kwargs["allow_websocket_origin"] = _DASHBOARD_ALLOW_ORIGIN

    for key, value in (("load_if_exists", True), ("use_basic_auth", False), ("debug", False)):
        if key in params:
            kwargs.setdefault(key, value)

    config_kwargs: Dict[str, object] = {"host": host, "port": port, "log_level": "info"}
    config_params = inspect.signature(uvicorn.Config).parameters
    if "loop" in config_params:
        config_kwargs.setdefault("loop", "asyncio")
    if "lifespan" in config_params:
        config_kwargs.setdefault("lifespan", "auto")

    process = multiprocessing.Process(
        target=_dashboard_process_create_app,
        name="optuna-dashboard",
        args=(storage_url, kwargs, config_kwargs),
        daemon=True,
    )
    process.start()

    def _stop() -> None:
        if not process.is_alive():
            return
        process.terminate()
        process.join(timeout=5)
        if process.is_alive():
            logger.warning("optuna-dashboard process did not terminate cleanly; killing")
            process.kill()
            process.join(timeout=2)
        close = getattr(process, "close", None)
        if callable(close):
            close()

    return _DashboardServer(
        storage=storage_url,
        study_name=study_name,
        host=host,
        port=port,
        stop=_stop,
        process=process,
    )


def _start_dashboard_server_legacy(
    storage_url: str, study_name: Optional[str], host: str, port: int
) -> Optional[_DashboardServer]:
    args, kwargs, resolved_port = _build_run_server_call(storage_url, study_name, host, port)
    if not args:
        logger.error("Unable to prepare optuna-dashboard run_server call; dashboard disabled")
        return None

    process = multiprocessing.Process(
        target=_dashboard_process_run_server,
        name="optuna-dashboard",
        args=(storage_url, args, kwargs),
        daemon=True,
    )
    process.start()

    def _stop() -> None:
        if not process.is_alive():
            return
        process.terminate()
        process.join(timeout=5)
        if process.is_alive():
            logger.warning("optuna-dashboard process did not terminate cleanly; killing")
            process.kill()
            process.join(timeout=2)
        close = getattr(process, "close", None)
        if callable(close):
            close()

    final_port = resolved_port if resolved_port is not None else port
    return _DashboardServer(
        storage=storage_url,
        study_name=study_name,
        host=host,
        port=final_port,
        stop=_stop,
        process=process,
    )


def _build_run_server_call(storage_url: str, study_name: Optional[str], host: str, port: int):
    if not _dashboard_available or optuna_dashboard is None:
        return [], {}, None

    run_server = getattr(optuna_dashboard, "run_server", None)
    if run_server is None:
        return [], {}, None

    params = inspect.signature(run_server).parameters
    args = [storage_url]
    kwargs: Dict[str, object] = {}

    if study_name:
        if "study_names" in params:
            kwargs["study_names"] = [study_name]
        elif "study_name" in params:
            kwargs["study_name"] = study_name

    if "host" in params:
        kwargs["host"] = host

    resolved_port: Optional[int] = None
    if "port" in params:
        kwargs["port"] = port
        resolved_port = port
    else:
        param = params.get("port")
        if param is not None and param.default is not inspect._empty and isinstance(param.default, int):
            resolved_port = param.default

    if resolved_port is None:
        resolved_port = port

    if "path" in params:
        kwargs["path"] = _DASHBOARD_PATH

    if _DASHBOARD_ALLOW_ORIGIN and "allow_websocket_origin" in params:
        kwargs["allow_websocket_origin"] = _DASHBOARD_ALLOW_ORIGIN

    if "load_if_exists" in params and "load_if_exists" not in kwargs:
        kwargs["load_if_exists"] = True

    if "debug" in params and "debug" not in kwargs:
        kwargs["debug"] = False

    if "use_basic_auth" in params and "use_basic_auth" not in kwargs:
        kwargs["use_basic_auth"] = False

    return args, kwargs, resolved_port


def _ensure_dashboard_server(storage_path: str, study_name: Optional[str]) -> Optional[_DashboardServer]:
    if not _dashboard_available or optuna_dashboard is None:
        return None

    storage_abs = os.path.abspath(storage_path)
    storage_url = f"sqlite:///{storage_abs}"

    global _dashboard_server

    with _dashboard_lock:
        if _dashboard_server and _dashboard_server.is_alive():
            if (
                _dashboard_server.storage == storage_url
                and _dashboard_server.study_name == study_name
                and _dashboard_server.host == _DASHBOARD_BIND_HOST
            ):
                return _dashboard_server
            _stop_dashboard_server_locked()
        elif _dashboard_server:
            _stop_dashboard_server_locked()

        server = _start_dashboard_server(storage_url, study_name, _DASHBOARD_BIND_HOST, _DASHBOARD_PORT)
        if server is None:
            return None

        _dashboard_server = server
        return _dashboard_server


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


def _resolve_study_dir(study_name: str) -> Optional[str]:
    """Return an absolute study directory within STUDY_ROOT or None when invalid."""

    if not study_name:
        return None

    normalized = os.path.normpath(study_name)
    if normalized in (".", "") or normalized.startswith(".."):
        return None

    candidate = os.path.abspath(os.path.join(STUDY_ROOT, normalized))
    root = os.path.abspath(STUDY_ROOT)

    try:
        common = os.path.commonpath([candidate, root])
    except ValueError:
        # Triggered when paths are on different drives under Windows semantics.
        return None

    if common != root:
        return None

    return candidate


def _sanitize_study_name(raw: Optional[str]) -> Optional[str]:
    """Return a filesystem-friendly study name or None when unusable."""

    if raw is None:
        return None

    cleaned = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in raw.strip())
    cleaned = cleaned.strip("._-")
    return cleaned or None


def _display_study_name(meta: Dict[str, object]) -> Optional[str]:
    """Determine the best study name to display to users."""

    requested = meta.get("requested_study_name")
    if isinstance(requested, str) and requested.strip():
        return requested.strip()

    fallback = meta.get("study_name")
    if isinstance(fallback, str) and fallback.strip():
        return fallback.strip()

    return None


def _optuna_db_response(study_name: str):
    """Return the Optuna DB for the study or an error response when missing."""

    study_dir = _resolve_study_dir(study_name)
    if study_dir is None:
        return JSONResponse({"error": "invalid study name"}, status_code=400)

    db_path = os.path.join(study_dir, "optuna_db.db")
    if not os.path.isfile(db_path):
        return JSONResponse({"error": "optuna database not found"}, status_code=404)

    safe_study_name = study_name.replace("/", "_").replace("\\", "_")
    download_name = f"{safe_study_name or 'study'}_optuna_db.db"

    return FileResponse(db_path, filename=download_name)


def _resolve_job_study_name(job: Job) -> Optional[str]:
    """Best effort resolution of the study name associated with a job."""

    request_meta = _read_study_request(job.id)
    meta_name = request_meta.get("study_name")
    if isinstance(meta_name, str) and meta_name.strip():
        return meta_name

    if job.result_dir:
        try:
            root = os.path.abspath(STUDY_ROOT)
            candidate = os.path.abspath(job.result_dir)
            common = os.path.commonpath([candidate, root])
        except ValueError:
            return None

        if common != root:
            return None

        rel = os.path.relpath(candidate, root)
        if rel not in (".", ""):
            return rel

    return None


@app.get("/")
async def index(request: Request):
    """Render the upload form with study argument defaults sourced from environment variables."""
    study_arg_specs = [
        {
            "key": key,
            "label": spec.label,
            "kind": spec.kind,
            "default": spec.default,
            "minimum": spec.minimum,
            "default_text": f"{spec.default:g}" if isinstance(spec.default, float) else str(spec.default),
            "step": "1" if spec.kind == "int" else "any",
        }
        for key, spec in _STUDY_ARG_SPECS.items()
    ]
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "study_arg_specs": study_arg_specs,
        },
    )


def _coerce_upload(value: Optional[UploadFile | Sequence[UploadFile]]) -> Optional[UploadFile]:
    """Return the first actual upload object or ``None`` when the field was absent."""

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
    file: Optional[UploadFile | Sequence[UploadFile]] = File(None),
    qubo_matrix: Optional[UploadFile | Sequence[UploadFile]] = File(None),
    study_name: Optional[str] = Form(None),
    study_args_raw: Optional[str] = Form(None, alias="study_args"),
    energy_shift: Optional[str] = Form("0"),
    h_vector: Optional[UploadFile | Sequence[UploadFile]] = File(None),
    g_vector: Optional[UploadFile | Sequence[UploadFile]] = File(None),
):
    """Handle problem uploads, supporting both direct Ising inputs and derived QUBO conversions."""
    # Normalise optional inputs that may arrive as lists or other sentinel values
    file = _coerce_upload(file)
    qubo_matrix = _coerce_upload(qubo_matrix)
    h_vector = _coerce_upload(h_vector)
    g_vector = _coerce_upload(g_vector)

    if file is None and qubo_matrix is None:
        return JSONResponse({"error": "Upload either a J matrix or a QUBO matrix"}, status_code=400)

    if file is not None and qubo_matrix is not None:
        return JSONResponse({"error": "Provide only one of J matrix or QUBO matrix"}, status_code=400)

    if file is not None and not file.filename.endswith(".npy"):
        return JSONResponse({"error": "Only .npy files accepted for the J matrix"}, status_code=400)

    if qubo_matrix is not None and not qubo_matrix.filename.endswith(".npy"):
        return JSONResponse({"error": "QUBO matrix must be a .npy file"}, status_code=400)

    if qubo_matrix is not None and h_vector is not None:
        return JSONResponse(
            {"error": "Do not include an h_vector when uploading a QUBO matrix; it is derived automatically."},
            status_code=400,
        )

    for optional_file, field_name in ((h_vector, "h_vector"), (g_vector, "g_vector")):
        if optional_file is None:
            continue
        if not optional_file.filename.endswith(".npy"):
            return JSONResponse({"error": f"{field_name} must be a .npy file"}, status_code=400)

    try:
        parsed_study_args = _parse_study_args(study_args_raw)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)

    try:
        energy_shift_value = float(energy_shift) if energy_shift not in (None, "") else 0.0
    except (TypeError, ValueError):
        return JSONResponse({"error": "energy_shift must be numeric"}, status_code=400)

    job_id = str(uuid.uuid4())
    job_dir = os.path.join(UPLOAD_ROOT, job_id)

    requested_study_name = (study_name or "").strip()
    sanitised_study_name = _sanitize_study_name(requested_study_name) if requested_study_name else None

    if requested_study_name and sanitised_study_name is None:
        return JSONResponse(
            {"error": "study name must include letters, numbers, dashes, dots or underscores"}, status_code=400
        )

    if sanitised_study_name:
        study_dir_candidate = _resolve_study_dir(sanitised_study_name)
        if study_dir_candidate is None:
            return JSONResponse({"error": "invalid study name"}, status_code=400)
        if os.path.exists(study_dir_candidate):
            return JSONResponse({"error": "study name already exists; choose a different name"}, status_code=409)
        study_name_to_use = sanitised_study_name
        study_dir = study_dir_candidate
    else:
        requested_study_name = None
        study_name_to_use = job_id
        study_dir = os.path.join(STUDY_ROOT, study_name_to_use)

    try:
        os.makedirs(study_dir, exist_ok=False)
    except FileExistsError:
        return JSONResponse({"error": "study name already exists; choose a different name"}, status_code=409)

    os.makedirs(job_dir, exist_ok=True)

    source_filename: str
    input_format: str

    if qubo_matrix is not None:
        try:
            qubo_matrix.file.seek(0)
            raw_bytes = qubo_matrix.file.read()
        except Exception as exc:
            return JSONResponse({"error": f"failed to read QUBO upload: {exc}"}, status_code=500)

        try:
            qubo_array = np.load(io.BytesIO(raw_bytes), allow_pickle=False)
        except Exception as exc:
            return JSONResponse({"error": f"failed to load QUBO matrix: {exc}"}, status_code=400)

        try:
            j_matrix, derived_h, derived_shift = QUBO_to_Ising(qubo_array)
        except AssertionError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        except Exception as exc:
            return JSONResponse({"error": f"failed to convert QUBO matrix: {exc}"}, status_code=400)

        try:
            np.save(os.path.join(job_dir, "J.npy"), j_matrix)
            np.save(os.path.join(job_dir, "h_vector.npy"), derived_h)
            with open(os.path.join(job_dir, "qubo_matrix.npy"), "wb") as out:
                out.write(raw_bytes)
        except Exception as exc:
            return JSONResponse({"error": f"failed to persist derived matrices: {exc}"}, status_code=500)

        source_filename = qubo_matrix.filename
        input_format = "qubo"
        energy_shift_value = float(derived_shift)
    else:
        assert file is not None  # for type checkers
        dest = os.path.join(job_dir, "J.npy")
        try:
            file.file.seek(0)
            with open(dest, "wb") as out:
                shutil.copyfileobj(file.file, out)
        except Exception as exc:
            return JSONResponse({"error": f"failed to save file: {exc}"}, status_code=500)

        if h_vector is not None:
            try:
                h_vector.file.seek(0)
                with open(os.path.join(job_dir, "h_vector.npy"), "wb") as out:
                    shutil.copyfileobj(h_vector.file, out)
            except Exception as exc:
                return JSONResponse({"error": f"failed to save vector file: {exc}"}, status_code=500)

        source_filename = file.filename
        input_format = "ising"

    if g_vector is not None:
        try:
            g_vector.file.seek(0)
            with open(os.path.join(job_dir, "g_vector.npy"), "wb") as out:
                shutil.copyfileobj(g_vector.file, out)
        except Exception as exc:
            return JSONResponse({"error": f"failed to save vector file: {exc}"}, status_code=500)

    study_payload: Dict[str, object] = {
        "study_name": study_name_to_use,
        "input_format": input_format,
        "energy_shift": energy_shift_value,
    }
    if parsed_study_args:
        study_payload["study_args"] = parsed_study_args
    if requested_study_name:
        study_payload["requested_study_name"] = requested_study_name

    try:
        with open(os.path.join(job_dir, REQUEST_FILENAME), "w", encoding="utf-8") as f:
            json.dump(study_payload, f)
    except Exception as e:
        return JSONResponse({"error": f"failed to save study request: {e}"}, status_code=500)

    job = Job(id=job_id, filename=source_filename, status=JobStatus.QUEUED, result_dir=study_dir)
    with SessionLocal() as db:
        db.add(job)
        db.commit()
        status = job.status.value

    return {"job_id": job_id, "status": status}


@app.get("/jobs")
def list_jobs():
    """Return all jobs ordered by submission time (newest first)."""
    with SessionLocal() as db:
        jobs = db.query(Job).order_by(Job.created_at.desc()).all()
        payload = []
        for j in jobs:
            meta = _read_study_request(j.id)
            study_name = meta.get("study_name")
            if not isinstance(study_name, str) or not study_name.strip():
                study_name = None
            else:
                study_name = study_name.strip()
            payload.append(
                {
                    "id": j.id,
                    "status": j.status.value,
                    "filename": j.filename,
                    "study_name": study_name,
                    "study_display_name": _display_study_name(meta),
                    "created_at": j.created_at.isoformat() + "Z",
                    "updated_at": j.updated_at.isoformat() + "Z",
                    "error": j.error,
                }
            )
        return payload


@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    """Return detailed information for a single job, including derived study metadata."""
    with SessionLocal() as db:
        j = db.get(Job, job_id)
        if not j:
            return JSONResponse({"error": "not found"}, status_code=404)
        request_meta = _read_study_request(job_id)
        study_name = request_meta.get("study_name")
        if isinstance(study_name, str) and study_name.strip():
            study_name = study_name.strip()
        else:
            study_name = None
        return {
            "id": j.id,
            "status": j.status.value,
            "filename": j.filename,
            "study_name": study_name,
            "study_display_name": _display_study_name(request_meta),
            "created_at": j.created_at.isoformat() + "Z",
            "updated_at": j.updated_at.isoformat() + "Z",
            "error": j.error,
        }


@app.get("/jobs/{job_id}/download")
def download_results(job_id: str):
    """Stream a ZIP archive containing the persisted study artifacts for a completed job."""
    with SessionLocal() as db:
        j = db.get(Job, job_id)
        if not j:
            return JSONResponse({"error": "not found"}, status_code=404)
        if (j.status != JobStatus.DONE) and (j.status != JobStatus.RUNNING):
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


@app.get("/jobs/{job_id}/dashboard")
def job_dashboard(job_id: str, request: Request):
    """Ensure the Optuna dashboard is running and return the external URL to access it."""
    if not _dashboard_available:
        return JSONResponse({"error": "optuna-dashboard integration is unavailable"}, status_code=503)
    with SessionLocal() as db:
        j = db.get(Job, job_id)
        if not j:
            return JSONResponse({"error": "not found"}, status_code=404)
        if (j.status != JobStatus.DONE) and (j.status != JobStatus.RUNNING):
            return JSONResponse({"error": f"job not done (status={j.status.value})"}, status_code=400)
        study_name = _resolve_job_study_name(j)
        if not study_name:
            return JSONResponse({"error": "unable to resolve study"}, status_code=404)

    study_dir = _resolve_study_dir(study_name)
    if study_dir is None:
        return JSONResponse({"error": "invalid study name"}, status_code=400)

    db_path = os.path.join(study_dir, "optuna_db.db")
    if not os.path.isfile(db_path):
        return JSONResponse({"error": "optuna database not found"}, status_code=404)

    server = _ensure_dashboard_server(db_path, study_name)
    if server is None:
        return JSONResponse({"error": "failed to start optuna-dashboard server"}, status_code=503)

    forwarded_proto = request.headers.get("x-forwarded-proto")
    public_scheme = os.environ.get("OPTUNA_DASHBOARD_PUBLIC_SCHEME") or forwarded_proto or request.url.scheme

    forwarded_host = request.headers.get("x-forwarded-host")
    public_host = os.environ.get("OPTUNA_DASHBOARD_PUBLIC_HOST") or (
        forwarded_host.split(",")[0].strip() if forwarded_host else (request.url.hostname or "localhost")
    )

    public_port_env = os.environ.get("OPTUNA_DASHBOARD_PUBLIC_PORT")
    if public_port_env is None:
        public_port = str(server.port)
    else:
        public_port = public_port_env.strip()

    netloc = public_host
    if ":" not in netloc and public_port:
        netloc = f"{public_host}:{public_port}"

    path_fragment = "" if _DASHBOARD_PATH == "/" else _DASHBOARD_PATH
    storage_url = f"sqlite:///{os.path.abspath(db_path)}"
    query_params = {"storage": storage_url}
    if study_name:
        query_params["study"] = study_name
        query_params.setdefault("study_name", study_name)  # backward compatibility for older dashboards
    query_string = f"?{urllib.parse.urlencode(query_params)}"

    dashboard_url = f"{public_scheme}://{netloc}{path_fragment}{query_string}"
    return {"dashboard_url": dashboard_url}


@app.get("/jobs/{job_id}/optuna-db")
def download_job_optuna_db(job_id: str):
    with SessionLocal() as db:
        job = db.get(Job, job_id)
        if not job:
            return JSONResponse({"error": "not found"}, status_code=404)
        study_name = _resolve_job_study_name(job)

    if not study_name:
        return JSONResponse({"error": "unable to resolve study"}, status_code=404)

    return _optuna_db_response(study_name)
