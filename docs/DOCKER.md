# Docker Deployment Guide

This guide explains how to deploy the Catalog Enrichment application using Docker and Docker Compose.

## Architecture

The application consists of the following services:

- **Frontend** (Port 3000): Next.js UI for product catalog enrichment
- **Backend** (Port 8000): FastAPI backend for orchestrating enrichment workflows
- **Nemotron 3 Nano Omni VLM NIM** (Port 8001): Omni-modal model for image analysis
- **Nemotron 3.5 Lightning LLM NIM** (Port 8002): Large Language Model for text generation
- **Flux NIM** (Port 8003): Image generation model for product variations
- **Trellis NIM** (Port 8004): 3D asset generation model
- **Embeddings NIM** (Post 8005): Embeddings for policy compliance
- **Milvus Stack** (Ports 19530, 9091, 9001): Persistent vector search for loaded policy PDFs
- **External Search API**: Exa is used by the product web insights agent

## Prerequisites

- Docker 24.0+ with Docker Compose
- NVIDIA 4xH100 (80GB) GPU with Docker GPU support (nvidia-docker2)
- NVIDIA NGC API Key
- HuggingFace Token (for Flux model)
- Optional Exa API Key (for product web insights)
- 512GB disk space

## Setup

### 1. Environment Variables

Create a `.env` file in the project root:

```bash
# NVIDIA NGC API Key (required for all NIM services)
NGC_API_KEY=your_ngc_api_key_here

# HuggingFace Token (required for Flux model)
HF_TOKEN=your_huggingface_token_here

# Exa API Key (optional; without it, the Web Insights tab shows a disabled message)
EXA_API_KEY=your_exa_api_key_here

# Unique credentials required by both MinIO and Milvus
MINIO_ACCESS_KEY=your_unique_minio_access_key
MINIO_SECRET_KEY=your_unique_minio_secret_key

# Optional. Published ports bind to localhost when omitted.
# Only use 0.0.0.0 behind authentication, TLS, rate limiting, and network controls.
# CATALOG_BIND_ADDRESS=0.0.0.0
```

This repository is a reference blueprint and does not provide application-level
authentication or rate limiting. Its Compose files bind all published ports to
`127.0.0.1` by default. Do not expose the stack directly to an untrusted
network. See [Security Policy](../SECURITY.md) before changing the bind address.

### 2. Create Cache Directory

```bash
export LOCAL_NIM_CACHE=~/.cache/nim
mkdir -p "$LOCAL_NIM_CACHE"
chmod a+w "$LOCAL_NIM_CACHE"
```

### 3. Create Shared Docker Network

```bash
docker network create catalog-network || true
```

## Running the Application

### Start All Services

```bash
docker-compose up -d
docker compose -f docker-compose.rag.yml up -d
```

### Start Specific Services

```bash
# Start only backend and frontend (without NIM models)
docker-compose up -d backend frontend

# Start a specific NIM model
docker-compose up -d vlm-nim

# Start all NIM models
docker-compose up -d vlm-nim llm-nim flux-nim trellis-nim

# Start the persistent policy RAG stack
docker compose -f docker-compose.rag.yml up -d
```

### View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f backend
docker-compose logs -f frontend
docker compose -f docker-compose.rag.yml logs -f milvus-standalone
```

### Stop Services

```bash
# Stop all services
docker-compose down

# Stop and remove volumes
docker-compose down -v
docker compose -f docker-compose.rag.yml down -v
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
- **Backend API through the gateway**: http://localhost:3000/api
- **Health Check through the gateway**: http://localhost:3000/api/health
- **Milvus gRPC**: localhost:19530
- **Milvus health**: localhost:9091
- **MinIO Console**: http://localhost:9001

The policy RAG stack requires unique MinIO credentials in `.env` and passes the
same values to Milvus. Compose refuses to start the stack when either value is
missing.

## GPU Configuration

The default configuration assigns one GPU to each NIM model:

- Nemotron 3 Nano Omni VLM: GPU 0
- Nemotron 3.5 Lightning LLM: GPU 1
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
docker compose -f docker-compose.rag.yml logs milvus-standalone
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
docker compose -f docker-compose.rag.yml down -v
```

### Clean Up Cache

```bash
rm -rf ~/.cache/nim/*
```
