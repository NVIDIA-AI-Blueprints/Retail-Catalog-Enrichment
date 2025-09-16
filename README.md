# Catalog Enrichment System

A GenAI-powered catalog enrichment system that transforms basic product images into comprehensive, rich catalog entries using NVIDIA's Nemotron VLM for content analysis, Nemotron LLM for intelligent prompt planning, and FLUX models for generating high-quality product variations.

## Development setup (uv + uvicorn)

This project uses [`uv`](https://docs.astral.sh/uv/) for Python environments and dependency management and `uvicorn` for serving the ASGI backend.

### Prerequisites
- Python 3.11+
- `uv` installed (macOS/Linux):
  - `curl -LsSf https://astral.sh/uv/install.sh | sh`

### Quick start (no explicit venv)
Run the backend directly using `uv` with an isolated environment:

```
uv run uvicorn --app-dir src backend.main:app --host 0.0.0.0 --port 8000 --reload
```

### Recommended: create a local virtual environment
```
# Create and activate a local venv managed by uv
uv venv .venv
source .venv/bin/activate

# Install project dependencies from pyproject.toml
uv pip install -e .

# Run the backend
uvicorn --app-dir src backend.main:app --host 0.0.0.0 --port 8000 --reload
```

Backend endpoints:
- `GET /` → plaintext greeting
- `GET /health` → JSON health status
- `POST /vlm/describe` → analyze product image and generate title, description, categories
- `POST /image/variation` → full enrichment pipeline with variation image generation

### API Endpoints

#### VLM: Describe an image

1) Set your NVIDIA API key in `.env`:

```
cp .env .env.local  # optional; or edit .env directly
echo "NVIDIA_API_KEY=YOUR_KEY_HERE" >> .env
```

2) Start the backend (see above), then call the endpoint:

```
curl -X POST \
  -F "image=@path/to/your_image.png;type=image/png" \
  http://localhost:8000/vlm/describe
```

Response:

```json
{
  "title": "Enhanced Product Title",
  "description": "Detailed product description",
  "categories": ["clothing", "casual"]
}
```

#### Image Variation Generation

For the full enrichment pipeline including variation image generation:

```
curl -X POST \
  -F "image=@path/to/your_image.png;type=image/png" \
  http://localhost:8000/image/variation
```

Response:

```json
{
  "title": "Enhanced Product Title",
  "description": "Detailed product description",
  "categories": ["clothing", "casual"],
  "artifact_id": "unique_artifact_id",
  "image_path": "path/to/generated/variation.jpg",
  "metadata_path": "path/to/metadata.json",
  "variation_plan": "AI-generated plan for image variations"
}
```

### Architecture

The system follows a multi-stage AI pipeline:

1. **VLM Analysis**: NVIDIA Nemotron VLM analyzes the input product image
2. **Prompt Planning**: NVIDIA Nemotron LLM creates optimized prompts for image generation
3. **Image Generation**: FLUX models generate high-quality product variations
4. **Asset Management**: Generated assets are organized and persisted with metadata

This architecture leverages NVIDIA's advanced AI models to ensure high-quality, consistent results for catalog enrichment.

For detailed architecture diagrams and system documentation, see:
- [`docs/architecture.md`](docs/architecture.md) - Comprehensive architecture with Mermaid diagrams
- [`docs/architecture-simple.txt`](docs/architecture-simple.txt) - Simple ASCII overview


