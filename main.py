"""
ACE-Step 1.5 FastAPI Server
A REST API server for music generation using ACE-Step 1.5 model.
"""

import os
import uuid
import asyncio
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# ACE-Step imports (will be available when acestep is installed)
from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music


# Configuration
PROJECT_ROOT = os.environ.get("ACESTEP_PROJECT_ROOT", "/app")
CHECKPOINT_DIR = os.environ.get("ACESTEP_CHECKPOINT_DIR", "/app/checkpoints")
OUTPUT_DIR = os.environ.get("ACESTEP_OUTPUT_DIR", "/app/outputs")
DEVICE = os.environ.get("ACESTEP_DEVICE", "cuda")
DIT_CONFIG = os.environ.get("ACESTEP_DIT_CONFIG", "acestep-v15-turbo")
LM_MODEL = os.environ.get("ACESTEP_LM_MODEL", "acestep-5Hz-lm-1.7B")
LM_BACKEND = os.environ.get("ACESTEP_LM_BACKEND", "vllm")

# Global handlers
dit_handler: Optional[AceStepHandler] = None
llm_handler: Optional[LLMHandler] = None

# Task storage
tasks: Dict[str, Dict[str, Any]] = {}


class TaskStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class AudioFormat(str, Enum):
    MP3 = "mp3"
    WAV = "wav"
    FLAC = "flac"


class TaskType(str, Enum):
    TEXT2MUSIC = "text2music"
    COVER = "cover"
    REPAINT = "repaint"
    LEGO = "lego"
    EXTRACT = "extract"
    COMPLETE = "complete"


# Request/Response Models
class GenerateRequest(BaseModel):
    caption: str = Field(default="", description="Music description prompt")
    lyrics: str = Field(default="", description="Lyrics content")
    instrumental: bool = Field(default=False, description="Generate instrumental music")
    
    # Music metadata
    bpm: Optional[int] = Field(default=None, ge=30, le=300, description="Tempo in BPM")
    key_scale: str = Field(default="", description="Musical key (e.g., 'C Major', 'Am')")
    time_signature: str = Field(default="", description="Time signature (2, 3, 4, or 6)")
    duration: float = Field(default=-1.0, ge=-1.0, le=600.0, description="Duration in seconds (10-600)")
    vocal_language: str = Field(default="en", description="Lyrics language code")
    
    # Generation parameters
    thinking: bool = Field(default=True, description="Use LM for audio code generation")
    inference_steps: int = Field(default=8, ge=1, le=200, description="Number of inference steps")
    guidance_scale: float = Field(default=7.0, ge=1.0, le=15.0, description="Guidance scale")
    seed: int = Field(default=-1, description="Random seed (-1 for random)")
    batch_size: int = Field(default=2, ge=1, le=8, description="Number of samples to generate")
    
    # Advanced parameters
    shift: float = Field(default=3.0, ge=1.0, le=5.0, description="Timestep shift factor")
    infer_method: str = Field(default="ode", description="Inference method (ode or sde)")
    audio_format: AudioFormat = Field(default=AudioFormat.MP3, description="Output audio format")
    
    # Task type
    task_type: TaskType = Field(default=TaskType.TEXT2MUSIC, description="Generation task type")
    
    # LM parameters
    lm_temperature: float = Field(default=0.85, ge=0.0, le=2.0)
    lm_top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    use_cot_metas: bool = Field(default=True)
    use_cot_caption: bool = Field(default=True)
    use_cot_lyrics: bool = Field(default=False)


class SampleRequest(BaseModel):
    query: str = Field(..., description="Natural language description for sample generation")
    instrumental: bool = Field(default=False)
    vocal_language: Optional[str] = Field(default=None)
    batch_size: int = Field(default=2, ge=1, le=8)
    duration: float = Field(default=-1.0)
    audio_format: AudioFormat = Field(default=AudioFormat.MP3)


class TaskResponse(BaseModel):
    task_id: str
    status: TaskStatus
    created_at: str
    message: str


class TaskStatusResponse(BaseModel):
    task_id: str
    status: TaskStatus
    created_at: str
    completed_at: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    dit_loaded: bool
    llm_loaded: bool
    device: str
    dit_config: str
    lm_model: str


class ModelsResponse(BaseModel):
    dit_models: List[str]
    lm_models: List[str]
    current_dit: str
    current_lm: str


def initialize_handlers():
    """Initialize the ACE-Step handlers."""
    global dit_handler, llm_handler
    
    dit_handler = AceStepHandler()
    llm_handler = LLMHandler()
    
    # Initialize DiT handler
    dit_handler.initialize_service(
        project_root=PROJECT_ROOT,
        config_path=DIT_CONFIG,
        device=DEVICE
    )
    
    # Initialize LLM handler
    llm_handler.initialize(
        checkpoint_dir=CHECKPOINT_DIR,
        lm_model_path=LM_MODEL,
        backend=LM_BACKEND,
        device=DEVICE
    )
    
    # Ensure output directory exists
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    # Startup
    initialize_handlers()
    yield
    # Shutdown
    pass


# Create FastAPI app
app = FastAPI(
    title="ACE-Step 1.5 API",
    description="REST API for ACE-Step 1.5 music generation model",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def run_generation_task(task_id: str, params: GenerationParams, config: GenerationConfig):
    """Background task for music generation."""
    try:
        tasks[task_id]["status"] = TaskStatus.RUNNING
        
        # Run generation in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: generate_music(
                dit_handler,
                llm_handler,
                params,
                config,
                save_dir=os.path.join(OUTPUT_DIR, task_id)
            )
        )
        
        if result.success:
            tasks[task_id]["status"] = TaskStatus.SUCCEEDED
            tasks[task_id]["result"] = {
                "audios": [
                    {
                        "path": audio["path"],
                        "key": audio["key"],
                        "seed": audio["params"].get("seed"),
                        "duration": audio["params"].get("duration"),
                    }
                    for audio in result.audios
                ],
                "time_costs": result.time_costs if hasattr(result, "time_costs") else None,
            }
        else:
            tasks[task_id]["status"] = TaskStatus.FAILED
            tasks[task_id]["error"] = result.error
            
    except Exception as e:
        tasks[task_id]["status"] = TaskStatus.FAILED
        tasks[task_id]["error"] = str(e)
    finally:
        tasks[task_id]["completed_at"] = datetime.utcnow().isoformat()


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if dit_handler and llm_handler else "initializing",
        dit_loaded=dit_handler is not None,
        llm_loaded=llm_handler is not None,
        device=DEVICE,
        dit_config=DIT_CONFIG,
        lm_model=LM_MODEL,
    )


@app.get("/v1/models", response_model=ModelsResponse)
async def list_models():
    """List available models."""
    return ModelsResponse(
        dit_models=[
            "acestep-v15-turbo",
            "acestep-v15-turbo-shift1",
            "acestep-v15-turbo-shift3",
            "acestep-v15-turbo-continuous",
            "acestep-v15-base",
            "acestep-v15-sft",
        ],
        lm_models=[
            "acestep-5Hz-lm-0.6B",
            "acestep-5Hz-lm-1.7B",
            "acestep-5Hz-lm-4B",
        ],
        current_dit=DIT_CONFIG,
        current_lm=LM_MODEL,
    )


@app.post("/v1/music/generate", response_model=TaskResponse)
async def create_generation_task(
    request: GenerateRequest,
    background_tasks: BackgroundTasks
):
    """Create a new music generation task."""
    if not dit_handler or not llm_handler:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    task_id = str(uuid.uuid4())
    created_at = datetime.utcnow().isoformat()
    
    # Build generation parameters
    params = GenerationParams(
        task_type=request.task_type.value,
        caption=request.caption,
        lyrics=request.lyrics,
        instrumental=request.instrumental,
        bpm=request.bpm,
        keyscale=request.key_scale,
        timesignature=request.time_signature,
        duration=request.duration,
        vocal_language=request.vocal_language,
        thinking=request.thinking,
        inference_steps=request.inference_steps,
        guidance_scale=request.guidance_scale,
        seed=request.seed,
        shift=request.shift,
        infer_method=request.infer_method,
        lm_temperature=request.lm_temperature,
        lm_top_p=request.lm_top_p,
        use_cot_metas=request.use_cot_metas,
        use_cot_caption=request.use_cot_caption,
        use_cot_lyrics=request.use_cot_lyrics,
    )
    
    config = GenerationConfig(
        batch_size=request.batch_size,
        use_random_seed=(request.seed == -1),
        seeds=[request.seed] if request.seed != -1 else None,
        audio_format=request.audio_format.value,
    )
    
    # Store task
    tasks[task_id] = {
        "status": TaskStatus.QUEUED,
        "created_at": created_at,
        "completed_at": None,
        "result": None,
        "error": None,
    }
    
    # Schedule background task
    background_tasks.add_task(run_generation_task, task_id, params, config)
    
    return TaskResponse(
        task_id=task_id,
        status=TaskStatus.QUEUED,
        created_at=created_at,
        message="Task queued for processing"
    )


@app.post("/v1/music/sample", response_model=TaskResponse)
async def create_sample_task(
    request: SampleRequest,
    background_tasks: BackgroundTasks
):
    """Create a sample generation task from natural language description."""
    if not dit_handler or not llm_handler:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    task_id = str(uuid.uuid4())
    created_at = datetime.utcnow().isoformat()
    
    # Store task
    tasks[task_id] = {
        "status": TaskStatus.QUEUED,
        "created_at": created_at,
        "completed_at": None,
        "result": None,
        "error": None,
    }
    
    async def run_sample_task():
        try:
            tasks[task_id]["status"] = TaskStatus.RUNNING
            
            loop = asyncio.get_event_loop()
            
            # First, create sample using LLM
            from acestep.inference import create_sample
            sample_result = await loop.run_in_executor(
                None,
                lambda: create_sample(
                    llm_handler,
                    query=request.query,
                    instrumental=request.instrumental,
                    vocal_language=request.vocal_language,
                )
            )
            
            if not sample_result.success:
                tasks[task_id]["status"] = TaskStatus.FAILED
                tasks[task_id]["error"] = sample_result.error
                return
            
            # Build generation parameters from sample
            params = GenerationParams(
                caption=sample_result.caption,
                lyrics=sample_result.lyrics,
                instrumental=request.instrumental,
                bpm=sample_result.bpm,
                keyscale=sample_result.keyscale,
                timesignature=sample_result.timesignature,
                duration=request.duration if request.duration > 0 else sample_result.duration,
                vocal_language=sample_result.vocal_language,
                thinking=True,
            )
            
            config = GenerationConfig(
                batch_size=request.batch_size,
                audio_format=request.audio_format.value,
            )
            
            # Generate music
            result = await loop.run_in_executor(
                None,
                lambda: generate_music(
                    dit_handler,
                    llm_handler,
                    params,
                    config,
                    save_dir=os.path.join(OUTPUT_DIR, task_id)
                )
            )
            
            if result.success:
                tasks[task_id]["status"] = TaskStatus.SUCCEEDED
                tasks[task_id]["result"] = {
                    "sample": {
                        "caption": sample_result.caption,
                        "lyrics": sample_result.lyrics,
                        "bpm": sample_result.bpm,
                        "keyscale": sample_result.keyscale,
                        "timesignature": sample_result.timesignature,
                    },
                    "audios": [
                        {
                            "path": audio["path"],
                            "key": audio["key"],
                            "seed": audio["params"].get("seed"),
                        }
                        for audio in result.audios
                    ],
                }
            else:
                tasks[task_id]["status"] = TaskStatus.FAILED
                tasks[task_id]["error"] = result.error
                
        except Exception as e:
            tasks[task_id]["status"] = TaskStatus.FAILED
            tasks[task_id]["error"] = str(e)
        finally:
            tasks[task_id]["completed_at"] = datetime.utcnow().isoformat()
    
    background_tasks.add_task(run_sample_task)
    
    return TaskResponse(
        task_id=task_id,
        status=TaskStatus.QUEUED,
        created_at=created_at,
        message="Sample generation task queued"
    )


@app.get("/v1/tasks/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """Get the status of a generation task."""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = tasks[task_id]
    return TaskStatusResponse(
        task_id=task_id,
        status=task["status"],
        created_at=task["created_at"],
        completed_at=task["completed_at"],
        result=task["result"],
        error=task["error"],
    )


@app.get("/v1/audio/{task_id}/{filename}")
async def download_audio(task_id: str, filename: str):
    """Download a generated audio file."""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    file_path = Path(OUTPUT_DIR) / task_id / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    # Determine media type
    suffix = file_path.suffix.lower()
    media_types = {
        ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
        ".flac": "audio/flac",
    }
    media_type = media_types.get(suffix, "application/octet-stream")
    
    return FileResponse(
        path=str(file_path),
        media_type=media_type,
        filename=filename
    )


@app.delete("/v1/tasks/{task_id}")
async def delete_task(task_id: str):
    """Delete a task and its generated files."""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    # Remove files
    task_dir = Path(OUTPUT_DIR) / task_id
    if task_dir.exists():
        import shutil
        shutil.rmtree(task_dir)
    
    # Remove from storage
    del tasks[task_id]
    
    return {"message": "Task deleted successfully"}


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8000")),
        reload=os.environ.get("RELOAD", "false").lower() == "true",
    )
