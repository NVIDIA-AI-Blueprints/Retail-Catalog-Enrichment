# Retail Catalog Enrichment Blueprint

<div align="center">

![NVIDIA Logo](https://avatars.githubusercontent.com/u/178940881?s=200&v=4)

</div>

A GenAI-powered catalog enrichment system that transforms basic product images into comprehensive, rich catalog entries using NVIDIA's Nemotron VLM for content analysis, Nemotron LLM for intelligent prompt planning, FLUX models for generating high-quality product variations, and TRELLIS model for 3D asset generation.

## Architecture

![Shopping Assistant Diagram](deploy/diagram.jpeg)

## Key Features

- **AI-Powered Analysis**: NVIDIA Nemotron VLM for intelligent product understanding
- **Smart Categorization**: Automatic classification into predefined product categories
- **Intelligent Prompt Planning**: Context-aware image variation planning based on regional aesthetics
- **Multi-Language Support**: Generate product titles and descriptions in **10 regional locales**
- **Cultural Image Generation**: Create culturally-appropriate product backgrounds (Spanish courtyards, Mexican family spaces, British formal settings)
- **Quality Evaluation**: Automated VLM-based quality assessment of generated images with detailed scoring
- **3D Asset Generation**: Transform 2D product images into interactive 3D GLB models using Microsoft TRELLIS
- **Modular API**: Separate endpoints for VLM analysis, image generation, and 3D asset generation

## Documentation

- **[API Documentation](docs/API.md)** - Detailed API endpoints, parameters, and examples
- **[Docker Deployment Guide](docs/DOCKER.md)** - Docker and Docker Compose setup instructions
- **[Product Requirements (PRD)](PRD.md)** - Product requirements and feature specifications
- **[AI Agent Guidelines](AGENTS.md)** - Instructions for AI assistants working on this project

## Tech Stack

**Backend:**
- FastAPI + Uvicorn (ASGI server)
- Python 3.11+
- OpenAI client (NVIDIA endpoint)
- PIL (Pillow) for image processing

**Frontend:**
- Next.js 15 with React 19
- TypeScript
- Kaizen UI (KUI) design system
- Model-viewer for 3D assets

**AI Models:**
- NVIDIA Nemotron VLM (vision-language model)
- NVIDIA Nemotron LLM (prompt planning)
- FLUX models (image generation)
- Microsoft TRELLIS (3D generation)

**Infrastructure:**
- Docker & Docker Compose
- NVIDIA NIM containers
- HuggingFace model hosting

## Minimum System Requirements

### Hardware Requirements

Expect that you will want the NIM microservices to be self-hosted as you progress in your catalog enrichment pipeline development. For self-hosting the project with these microservices locally deployed, the recommended system requirement is **4 H100 GPUs** 

### Deployment Options

- Docker 28.0+
- Docker compose

## Quick Start

### Prerequisites

- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/) package manager
- NVIDIA API key for VLM/LLM services
- HuggingFace token for FLUX image generation

### Environment Setup

Create a `.env` file in the project root:

```bash
NGC_API_KEY=your_nvidia_api_key_here
HF_TOKEN=your_huggingface_token_here
```

**Getting API Keys:**
- NVIDIA API Key: [Get one here](https://build.nvidia.com/)
- HuggingFace Token: [Get one here](https://huggingface.co/settings/tokens)

### Local Development (Without Docker)

1. **Install uv** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Create and activate virtual environment**:
   ```bash
   uv venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   uv pip install -e .
   ```

4. **Run the backend**:
   ```bash
   uvicorn --app-dir src backend.main:app --host 0.0.0.0 --port 8000 --reload
   ```

5. **Run the frontend** (optional):
   ```bash
   cd src/ui
   pnpm install
   pnpm dev
   ```

The frontend at `http://localhost:3000`.

### Docker Deployment

For Docker deployment with all services (including NIM models), see the **[Docker Deployment Guide](docs/DOCKER.md)**.

**Quick Docker Start:**

1. **Create `.env` file** with required credentials:
   ```bash
   NGC_API_KEY=your_ngc_api_key_here
   HF_TOKEN=your_huggingface_token_here
   ```

2. **Create cache directories**:
   ```bash
   export LOCAL_NIM_CACHE=~/.cache/nim
   mkdir -p "$LOCAL_NIM_CACHE"
   chmod a+w "$LOCAL_NIM_CACHE"
   ```

3. **Start all services**:
   ```bash
   docker-compose up -d
   ```

4. **Access the application**:
   - Frontend: `http://localhost:3000`
   - Backend API: `http://localhost:8000`
   - Health Check: `http://localhost:8000/health`

## API Endpoints

The system provides three main endpoints:

- `POST /vlm/analyze` - Fast VLM/LLM analysis
- `POST /generate/variation` - Image generation with FLUX
- `POST /generate/3d` - 3D asset generation with TRELLIS

For detailed API documentation with request/response examples, see **[API Documentation](docs/API.md)**.

## License

This project will download and install additional third-party open source software projects. Review the license terms of these open source projects before use.