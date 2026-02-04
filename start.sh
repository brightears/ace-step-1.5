#!/bin/bash
# Start both ACE-Step API server and Gradio UI

# Start Gradio UI in background on port 7860
acestep --server-name 0.0.0.0 --port 7860 --init_service true &

# Start API server on port 8000 (background)
acestep-api --host 0.0.0.0 --port 8000 &

# Keep container running for RunPod web terminal
sleep infinity
