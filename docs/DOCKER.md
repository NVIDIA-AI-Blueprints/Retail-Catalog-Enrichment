# Docker Deployment Guide

This guide explains how to deploy the Catalog Enrichment application using Docker and Docker Compose.

## Architecture

The application consists of the following services:

- **Frontend** (Port 3000): Next.js UI for product catalog enrichment
- **Backend** (Port 8000): FastAPI backend for orchestrating enrichment workflows
- **VLM NIM** (Port 8001): Vision-Language Model for image analysis
- **LLM NIM** (Port 8002): Large Language Model for text generation
- **Flux NIM** (Port 8003): Image generation model for product variations
- **Trellis NIM** (Port 8004): 3D asset generation model

## Prerequisites

- Docker 24.0+ with Docker Compose
- NVIDIA GPU with Docker GPU support (nvidia-docker2)
- NVIDIA NGC API Key
- HuggingFace Token (for Flux model)
- At least 4 GPUs (or adjust GPU assignments in docker-compose.yml)

## Setup

### 1. Environment Variables

Create a `.env` file in the project root:

```bash
# NVIDIA NGC API Key (required for all NIM services)
NGC_API_KEY=your_ngc_api_key_here

# HuggingFace Token (required for Flux model)
HF_TOKEN=your_huggingface_token_here
```

### 2. Create Cache Directory

```bash
export LOCAL_NIM_CACHE=~/.cache/nim
mkdir -p "$LOCAL_NIM_CACHE"
chmod a+w "$LOCAL_NIM_CACHE"
```

## Running the Application

### Start All Services

```bash
docker-compose up -d
```

### Start Specific Services

```bash
# Start only backend and frontend (without NIM models)
docker-compose up -d backend frontend

# Start a specific NIM model
docker-compose up -d vlm-nim

# Start all NIM models
docker-compose up -d vlm-nim llm-nim flux-nim trellis-nim
```

### View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f backend
docker-compose logs -f frontend
```

### Stop Services

```bash
# Stop all services
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

## Building Images

### Build Backend

```bash
docker build -f src/backend/Dockerfile -t catalog-enrichment-backend .
```

### Build Frontend

```bash
docker build -f src/ui/Dockerfile -t catalog-enrichment-frontend ./src/ui
```

### Rebuild All Services

```bash
docker-compose build
docker-compose up -d
```

## Accessing the Application

Once all services are running:

- **Frontend UI**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **Health Check**: http://localhost:8000/health

## GPU Configuration

The default configuration assigns one GPU to each NIM model:

- VLM: GPU 0
- LLM: GPU 1
- Trellis: GPU 2
- Flux: GPU 3

To adjust GPU assignments, edit the `device_ids` in `docker-compose.yml`:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          device_ids: ['0', '1']  # Use multiple GPUs
          capabilities: [gpu]
```

## Troubleshooting

### Check GPU Availability

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
```

### Check Service Status

```bash
docker-compose ps
```

### Inspect Service Logs

```bash
docker-compose logs backend
docker-compose logs vlm-nim
```

### Restart a Service

```bash
docker-compose restart backend
```

### Remove and Rebuild

```bash
docker-compose down
docker-compose build --no-cache backend
docker-compose up -d
```

## Cleanup

### Remove All Containers and Images

```bash
docker-compose down --rmi all
```

### Clean Up Cache

```bash
rm -rf ~/.cache/nim/*
```

