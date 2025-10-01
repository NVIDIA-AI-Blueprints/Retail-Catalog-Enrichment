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
- `POST /vlm/describe` → augment existing product data with VLM-enhanced content (title, description, attributes, etc.)
- `POST /image/variation` → full enrichment pipeline with variation image generation

### API Endpoints

#### VLM: Augment Product Data with Multi-Language & Regional Support

1) Set your NVIDIA API key in `.env`:

```
cp .env .env.local  # optional; or edit .env directly
echo "NVIDIA_API_KEY=YOUR_KEY_HERE" >> .env
```

2) Start the backend (see above), then call the endpoint:

##### Image Only (Generation Mode):
```bash
curl -X POST \
  -F "image=@bag.jpg;type=image/jpeg" \
  -F "locale=en-US" \
  http://localhost:8000/vlm/describe
```

##### With Product Data (Augmentation Mode):
```bash
curl -X POST \
  -F "image=@bag.jpg;type=image/jpeg" \
  -F 'product_data={"title":"Classic Black Patent purse","description":"Elegant bag","price":15.99,"categories":["accessories","bag"],"tags":["bag","purse"]}' \
  -F "locale=en-US" \
  http://localhost:8000/vlm/describe
```

##### With Regional Localization (Spain Spanish):
```bash
curl -X POST \
  -F "image=@bag.jpg;type=image/jpeg" \
  -F 'product_data={"categories":["accessories","bag"],"title":"Black Purse","description":"Elegant bag","price":49.9,"tags":["bag"]}' \
  -F "locale=es-ES" \
  http://localhost:8000/vlm/describe
```

**Input Product Data Schema (optional):**
```json
{
  "title": "string",
  "description": "string",
  "price": "number",
  "categories": ["string"],
  "tags": ["string"]
}
```

**Response Schema:**

```json
{
  "title": "Glamorous Black Evening Handbag with Gold Accents",
  "description": "This exquisite handbag exudes sophistication and elegance. Crafted from high-quality, glossy leather, it boasts a sleek, rectangular shape with a slight taper, ensuring a chic silhouette that complements any outfit. The long, slender handles are designed for effortless elegance, while the gold-toned hardware, including buckles and zipper accents, adds a touch of opulence.",
  "categories": ["accessories"],
  "tags": ["black leather", "gold accents", "evening bag", "rectangular shape", "long handles", "sleek design", "sophisticated", "glamorous"],
  "colors": ["black", "gold"]
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


