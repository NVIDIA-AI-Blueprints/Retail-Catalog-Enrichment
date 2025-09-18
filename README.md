# Catalog Enrichment System

A GenAI-powered catalog enrichment system that transforms basic product images into comprehensive, rich catalog entries using NVIDIA's Nemotron VLM for content analysis, Nemotron LLM for intelligent prompt planning, and FLUX models for generating high-quality product variations.

## 🌍 Key Features

- **AI-Powered Analysis**: NVIDIA Nemotron VLM for intelligent product understanding
- **Smart Categorization**: Automatic classification into predefined product categories
- **Intelligent Prompt Planning**: Context-aware image variation planning based on regional aesthetics
- **Multi-Language Support**: Generate product titles and descriptions in **10 regional locales**
- **Cultural Image Generation**: Create culturally-appropriate product backgrounds (Spanish courtyards, Mexican family spaces, British formal settings)

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
- `POST /vlm/describe` → analyze product image and generate localized title, description, categories
- `POST /image/variation` → full enrichment pipeline with variation image generation

### API Endpoints

#### VLM: Describe an image with Multi-Language & Regional Support

1) Set your NVIDIA API key in `.env`:

```
cp .env .env.local  # optional; or edit .env directly
echo "NVIDIA_API_KEY=YOUR_KEY_HERE" >> .env
```

2) Start the backend (see above), then call the endpoint:

##### Basic Usage (defaults to en-US):
```
curl -X POST \
  -F "image=@path/to/your_image.png;type=image/png" \
  http://localhost:8000/vlm/describe
```

##### With Regional Localization:
```
curl -X POST \
  -F "image=@path/to/your_image.png;type=image/png" \
  -F "locale=es-ES" \
  http://localhost:8000/vlm/describe
```

Response:

```json
{
  "title": "Silla Ergonómica de Oficina",
  "description": "Silla de oficina cómoda con respaldo alto y soporte lumbar ajustable, perfecta para largas jornadas de trabajo.",
  "categories": ["furniture", "office"],
  "locale": "es-ES"
}
```

##### Supported Locales:

**English Variants:**
- `en-US` - American English (default)
- `en-GB` - British English  
- `en-AU` - Australian English
- `en-CA` - Canadian English

**Spanish Variants:**
- `es-ES` - Spain Spanish (uses "ordenador")
- `es-MX` - Mexican Spanish (uses "computadora") 
- `es-AR` - Argentinian Spanish
- `es-CO` - Colombian Spanish

**French Variants:**
- `fr-FR` - Metropolitan French
- `fr-CA` - Quebec French (Canadian)


