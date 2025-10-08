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
- `POST /vlm/analyze` → **Fast VLM analysis** (~2-5 seconds) - extract product fields without image generation
- `POST /generate/variation` → **Image generation with FLUX** (~30-60 seconds) - generate product variation using VLM results
- `POST /vlm/describe` → **Legacy complete pipeline** (~35-65 seconds) - VLM analysis + image generation in one call

### API Endpoints

#### Recommended Workflow: Split Pipeline for Better Performance

For optimal performance and flexibility, use the two-endpoint approach:

**1) Fast VLM Analysis (POST `/vlm/analyze`)** - Get product fields quickly (~2-5 seconds)
**2) Image Generation (POST `/generate/variation`)** - Generate variations on demand (~30-60 seconds)

This allows you to:
- Display product information immediately to users
- Generate images in the background or on-demand
- Cache VLM results and generate multiple variations
- Better error handling for each step

---

#### 1️⃣ Fast VLM Analysis: `/vlm/analyze`

Extract product fields using NVIDIA Nemotron VLM (no image generation).

**Setup:** Set your NVIDIA API key in `.env`:
```bash
echo "NVIDIA_API_KEY=YOUR_KEY_HERE" >> .env
```

**Usage Examples:**

##### Image Only (Generation Mode):
```bash
curl -X POST \
  -F "image=@bag.jpg;type=image/jpeg" \
  -F "locale=en-US" \
  http://localhost:8000/vlm/analyze
```

##### With Existing Product Data (Augmentation Mode):
```bash
curl -X POST \
  -F "image=@bag.jpg;type=image/jpeg" \
  -F 'product_data={"title":"Classic Black Patent purse","description":"Elegant bag","price":15.99,"categories":["accessories"],"tags":["bag","purse"]}' \
  -F "locale=en-US" \
  http://localhost:8000/vlm/analyze
```

##### Regional Localization (Spain Spanish):
```bash
curl -X POST \
  -F "image=@bag.jpg;type=image/jpeg" \
  -F 'product_data={"title":"Black Purse","description":"Elegant bag"}' \
  -F "locale=es-ES" \
  http://localhost:8000/vlm/analyze
```

**Input Schema (optional `product_data`):**
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
  "description": "This exquisite handbag exudes sophistication and elegance. Crafted from high-quality, glossy leather...",
  "categories": ["accessories"],
  "tags": ["black leather", "gold accents", "evening bag", "rectangular shape"],
  "colors": ["black", "gold"],
  "locale": "en-US"
}
```

---

#### 2️⃣ Image Generation: `/generate/variation`

Generate culturally-appropriate product variations using FLUX models based on VLM analysis results.

**Usage Example:**
```bash
# First, run VLM analysis (above) to get the fields, then:
curl -X POST \
  -F "image=@bag.jpg;type=image/jpeg" \
  -F "locale=en-US" \
  -F "title=Glamorous Black Evening Handbag with Gold Accents" \
  -F "description=This exquisite handbag exudes sophistication..." \
  -F 'categories=["accessories"]' \
  -F 'tags=["black leather","gold accents","evening bag"]' \
  -F 'colors=["black","gold"]' \
  http://localhost:8000/generate/variation
```

**Response Schema:**
```json
{
  "generated_image_b64": "iVBORw0KGgoAAAANS...",
  "artifact_id": "a4511bbed05242078f9e3f7ead3b2247",
  "image_path": "data/outputs/a4511bbed05242078f9e3f7ead3b2247.png",
  "metadata_path": "data/outputs/a4511bbed05242078f9e3f7ead3b2247.json",
  "locale": "en-US"
}
```

---

#### 3️⃣ Legacy Complete Pipeline: `/vlm/describe`

Complete enrichment pipeline (VLM analysis + image generation in one call).

**Note:** This endpoint is kept for backward compatibility but is slower (~35-65 seconds). Consider using the split approach above for better UX.

**Usage Example:**
```bash
curl -X POST \
  -F "image=@bag.jpg;type=image/jpeg" \
  -F "locale=en-US" \
  http://localhost:8000/vlm/describe
```

**Response Schema:**
```json
{
  "title": "Glamorous Black Evening Handbag with Gold Accents",
  "description": "This exquisite handbag exudes sophistication...",
  "categories": ["accessories"],
  "tags": ["black leather", "gold accents", "evening bag"],
  "colors": ["black", "gold"],
  "generated_image_b64": "iVBORw0KGgoAAAANS...",
  "locale": "en-US"
}
```

---

### Web UI

A Next.js-based web interface is available for interactive catalog enrichment.

**Setup:**
```bash
cd src/ui
pnpm install
pnpm dev
```

The UI will be available at `http://localhost:3000`

**Features:**
- **Drag & Drop Image Upload**: Upload product images for analysis
- **Locale Selector**: Choose from 10 supported regional locales using a dropdown menu
- **Reset Button**: One-click reset to clear all data and start fresh
- **Real-time VLM Analysis**: Get enriched product fields in ~2-5 seconds
- **Parallel Image Generation**: Generate 3 product variations simultaneously
- **Live Progress Indicators**: Visual feedback during analysis and generation
- **Product Field Augmentation**: View original and AI-augmented product data side-by-side

**Locale Support:**
The UI includes a dropdown selector with all supported locales:
- **English**: US, UK, Australia, Canada
- **Spanish**: Spain, Mexico, Argentina, Colombia
- **French**: France, Canada

The selected locale affects:
- Language and terminology in product descriptions
- Cultural context for generated image backgrounds
- Regional preferences (e.g., "ordenador" vs "computadora" for Spanish regions)

---

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


