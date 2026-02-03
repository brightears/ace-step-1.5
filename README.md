# ACE-Step 1.5 FastAPI Server

A REST API server for music generation using the ACE-Step 1.5 model.

## Features

- **FastAPI + Uvicorn** for high-performance async API
- **Async task queue** for background music generation
- **Multi-stage Docker build** with models baked in (~15GB image)
- **GPU support** via NVIDIA CUDA runtime

## Quick Start

### Prerequisites

- Docker with NVIDIA Container Toolkit
- NVIDIA GPU with CUDA support (4GB+ VRAM recommended)

### 1. Build the Docker Image

The Docker image includes all ACE-Step models (~15GB total). Build time will depend on your internet connection.

```bash
docker build -t acestep-api:latest .
```

### 2. Run with Docker Compose

```bash
docker compose up -d
```

The API will be available at `http://localhost:8000`.

> **Note:** Models are baked into the image, so no additional downloads are needed at runtime.

### 3. Run Locally (Development)

```bash
# Install dependencies
pip install -r requirements.txt

# Install ACE-Step
pip install git+https://github.com/ace-step/ACE-Step-1.5.git

# Run server
python main.py
```

## API Endpoints

### Health Check
```
GET /health
```

### List Models
```
GET /v1/models
```

### Generate Music
```
POST /v1/music/generate
```

**Request Body:**
```json
{
  "caption": "upbeat electronic dance music with heavy bass",
  "lyrics": "[Verse]\nDancing through the night...",
  "bpm": 128,
  "duration": 60,
  "batch_size": 2,
  "audio_format": "mp3"
}
```

### Generate from Description (Sample Mode)
```
POST /v1/music/sample
```

**Request Body:**
```json
{
  "query": "a soft Bengali love song",
  "instrumental": false,
  "batch_size": 2
}
```

### Get Task Status
```
GET /v1/tasks/{task_id}
```

### Download Audio
```
GET /v1/audio/{task_id}/{filename}
```

### Delete Task
```
DELETE /v1/tasks/{task_id}
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ACESTEP_PROJECT_ROOT` | `/app` | Project root directory |
| `ACESTEP_CHECKPOINT_DIR` | `/app/checkpoints` | Model checkpoints directory |
| `ACESTEP_OUTPUT_DIR` | `/app/outputs` | Generated audio output directory |
| `ACESTEP_DEVICE` | `cuda` | Device (cuda, cpu, mps) |
| `ACESTEP_DIT_CONFIG` | `acestep-v15-turbo` | DiT model configuration |
| `ACESTEP_LM_MODEL` | `acestep-5Hz-lm-1.7B` | Language model path |
| `ACESTEP_LM_BACKEND` | `vllm` | LLM backend (vllm, transformers) |
| `HOST` | `0.0.0.0` | Server host |
| `PORT` | `8000` | Server port |

## Example Usage

```python
import requests
import time

BASE_URL = "http://localhost:8000"

# Create generation task
response = requests.post(f"{BASE_URL}/v1/music/generate", json={
    "caption": "relaxing piano music with soft strings",
    "duration": 30,
    "batch_size": 2,
})
task = response.json()
task_id = task["task_id"]

# Poll for completion
while True:
    status = requests.get(f"{BASE_URL}/v1/tasks/{task_id}").json()
    if status["status"] in ["succeeded", "failed"]:
        break
    time.sleep(2)

# Download audio
if status["status"] == "succeeded":
    for audio in status["result"]["audios"]:
        filename = audio["path"].split("/")[-1]
        audio_data = requests.get(f"{BASE_URL}/v1/audio/{task_id}/{filename}")
        with open(filename, "wb") as f:
            f.write(audio_data.content)
```

## License

See the [ACE-Step 1.5 repository](https://github.com/ace-step/ACE-Step-1.5) for license information.
