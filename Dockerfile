# =============================================================================
# ACE-Step 1.5 FastAPI Server - Multi-stage Dockerfile
# =============================================================================
# This image includes the ACE-Step models (~15GB total)
# Build with: docker build -t acestep-api:latest .
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Builder - Install dependencies and build wheels
# -----------------------------------------------------------------------------
FROM python:3.11-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for faster dependency resolution
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Copy requirements first for better caching
COPY requirements.txt .

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install PyTorch with CUDA support
RUN pip install --no-cache-dir \
    torch==2.2.0 \
    torchaudio==2.2.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Install other requirements
RUN pip install --no-cache-dir -r requirements.txt

# Clone and install ACE-Step
RUN git clone https://github.com/ace-step/ACE-Step-1.5.git /tmp/acestep && \
    cd /tmp/acestep && \
    pip install --no-cache-dir -e . && \
    rm -rf /tmp/acestep/.git

# -----------------------------------------------------------------------------
# Stage 2: Model Downloader - Download models from HuggingFace
# -----------------------------------------------------------------------------
FROM python:3.11-slim as model-downloader

WORKDIR /models

# Install huggingface-cli
RUN pip install --no-cache-dir huggingface-hub

# Download main model package (includes VAE, Qwen3-Embedding, acestep-v15-turbo, acestep-5Hz-lm-1.7B)
RUN huggingface-cli download ACE-Step/Ace-Step1.5 --local-dir /models/checkpoints

# Optional: Download additional LM models (uncomment if needed)
# RUN huggingface-cli download ACE-Step/acestep-5Hz-lm-0.6B --local-dir /models/checkpoints/acestep-5Hz-lm-0.6B
# RUN huggingface-cli download ACE-Step/acestep-5Hz-lm-4B --local-dir /models/checkpoints/acestep-5Hz-lm-4B

# Optional: Download additional DiT models (uncomment if needed)
# RUN huggingface-cli download ACE-Step/acestep-v15-turbo-shift3 --local-dir /models/checkpoints/acestep-v15-turbo-shift3

# -----------------------------------------------------------------------------
# Stage 3: Runtime - Minimal image for running the application
# -----------------------------------------------------------------------------
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04 as runtime

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PATH="/opt/venv/bin:$PATH" \
    # ACE-Step configuration
    ACESTEP_PROJECT_ROOT=/app \
    ACESTEP_CHECKPOINT_DIR=/app/checkpoints \
    ACESTEP_OUTPUT_DIR=/app/outputs \
    ACESTEP_DEVICE=cuda \
    ACESTEP_DIT_CONFIG=acestep-v15-turbo \
    ACESTEP_LM_MODEL=acestep-5Hz-lm-1.7B \
    ACESTEP_LM_BACKEND=vllm \
    # Server configuration
    HOST=0.0.0.0 \
    PORT=8000

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Copy models from model-downloader stage
COPY --from=model-downloader /models/checkpoints /app/checkpoints

# Copy application code
COPY main.py .

# Create output directory
RUN mkdir -p /app/outputs

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run the application
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
