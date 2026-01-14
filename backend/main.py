import logging
from typing import List, Optional, Tuple
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .service import moss_service

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="MOSS API", version="0.1.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Static Files
# We serve the specific output directory for results, and data directory for inputs
# Input Data
app.mount("/files/data", StaticFiles(directory=moss_service.data_dir), name="data")
# Output Results
app.mount(
    "/files/output", StaticFiles(directory=moss_service.output_dir), name="output"
)


# --- Schemas ---
class ResourceResponse(BaseModel):
    images: List[str]
    audio: List[str]


class OptimizeRequest(BaseModel):
    image_path: str
    audio_path: str
    mode: str = "single"  # "single" or "pareto"
    weights: Optional[Tuple[float, float]] = None  # (Visual, Audio)
    seed_task_id: Optional[str] = None
    seed_index: int = 0


class TaskResponse(BaseModel):
    task_id: str
    status: str
    progress: float
    result_path: Optional[str] = None
    result_metrics: Optional[List[float] | List[List[float]]] = (
        None  # [Loss_Vis, Loss_Aud]
    )
    error: Optional[str] = None
    mode: str = "single"


# --- Endpoints ---


@app.get("/resources", response_model=ResourceResponse)
async def list_resources():
    """List available images and audio files in the data directory."""
    data_dir = moss_service.data_dir

    images = []
    audio = []

    # Simple recursive search or just top level?
    # Let's do top level + 'src' folder if user put stuff there?
    # Usually data/images and data/audio or just mixed.
    # Let's scan everything in data root.

    for f in data_dir.rglob("*"):
        # ExSlude output and cache directories
        # data_dir is absolute. f is absolute.
        # Check if "output" or "cache" is a component relative to data_dir
        try:
            rel = f.relative_to(data_dir)
            # If the first part is output or cache, skip
            if rel.parts[0] in ["output", "cache", "temp_dashboard", "temp_verify"]:
                continue
        except ValueError:
            continue

        if f.is_file():
            # Return relative path from data_dir
            rel_path = str(rel)
            if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
                images.append(rel_path)
            elif f.suffix.lower() in [".wav", ".mp3", ".flac"]:
                audio.append(rel_path)

    return {"images": images, "audio": audio}


@app.post("/optimize", response_model=TaskResponse)
async def start_optimization(req: OptimizeRequest, background_tasks: BackgroundTasks):
    """Start an optimization task."""

    # Resolve full paths
    img_full = moss_service.data_dir / req.image_path
    aud_full = moss_service.data_dir / req.audio_path

    if not img_full.exists():
        raise HTTPException(404, f"Image not found: {req.image_path}")
    if not aud_full.exists():
        raise HTTPException(404, f"Audio not found: {req.audio_path}")

    # We call the service synchronously for now as per `service.py` impl
    # but wrapper is `start_optimization` which returns ID.
    # If we want async, we should use background_tasks.add_task
    # But `moss_service.start_optimization` is currently blocking.
    # To make it non-blocking, we need to defer `_run_optimization` to background task.
    # Let's modify `start_optimization` to accept background_tasks or just run it here.

    # Better: generate ID here, set status pending, add task to background.
    import uuid
    import time

    task_id = str(uuid.uuid4())
    moss_service.tasks[task_id] = {
        "status": "pending",
        "progress": 0.0,
        "created_at": time.time(),
        "mode": req.mode,
        "params": req.dict(),
    }

    # Handle Seeding
    seed_mask = None
    if req.seed_task_id:
        try:
            # Load result from seed task
            # output_dir / seed_task_id / X.npy
            # We need to verify if task exists and has result?
            # For simplicity, check file existence.
            seed_dir = moss_service.output_dir / req.seed_task_id
            x_path = seed_dir / "X.npy"
            if x_path.exists():
                import numpy as np

                X = np.load(x_path)  # (Pop, Param)
                if 0 <= req.seed_index < len(X):
                    seed_mask = X[req.seed_index]
                    logger.info(
                        f"Loaded seed mask from {req.seed_task_id} index {req.seed_index}"
                    )
                else:
                    logger.warning("Seed index out of bounds")
            else:
                logger.warning(f"Seed task result not found: {x_path}")
        except Exception as e:
            logger.error(f"Seeding failed: {e}")

    background_tasks.add_task(
        moss_service._run_optimization,
        task_id,
        str(img_full),
        str(aud_full),
        req.mode,
        req.weights,
        seed_mask,
    )

    return {"task_id": task_id, "status": "pending", "progress": 0.0}


@app.get("/status/{task_id}", response_model=TaskResponse)
async def get_status(task_id: str):
    if task_id not in moss_service.tasks:
        raise HTTPException(404, "Task not found")

    task = moss_service.tasks[task_id]

    # Construct web-accessible result URL if completed
    result_path = None
    if task["status"] == "completed":
        # task["result_path"] is absolute file path.
        # We need relative URL.
        # It is in output_dir / task_id.
        # Mount is /files/output
        result_path = f"/files/output/{task_id}"

    return {
        "task_id": task_id,
        "status": task["status"],
        "progress": task["progress"],
        "result_path": result_path,
        "result_metrics": task.get("result_metrics"),
        "error": task.get("error"),
        "mode": task.get("mode", "single"),
    }


@app.get("/results/{task_id}/{index}/spectrogram")
async def get_individual_spectrogram(task_id: str, index: int):
    """
    Returns the spectrogram for a specific individual in a Pareto task.
    """
    try:
        buf = moss_service.get_individual_media(task_id, index, "spectrogram")
        return StreamingResponse(buf, media_type="image/png")
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        logger.error(f"Error serving spectrogram: {e}")
        raise HTTPException(500, "Internal Server Error")


@app.get("/results/{task_id}/{index}/audio")
async def get_individual_audio(task_id: str, index: int):
    """
    Returns the audio for a specific individual in a Pareto task.
    """
    try:
        buf = moss_service.get_individual_media(task_id, index, "audio")
        # Ensure correct headers for playback
        return StreamingResponse(buf, media_type="audio/wav")
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        logger.error(f"Error serving audio: {e}")
        raise HTTPException(500, "Internal Server Error")
