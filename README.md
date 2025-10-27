# Catalog Enrichment System

A GenAI-powered catalog enrichment system that transforms basic product images into comprehensive, rich catalog entries using NVIDIA's Nemotron VLM for content analysis, Nemotron LLM for intelligent prompt planning, FLUX models for generating high-quality product variations, and Microsoft's TRELLIS model for 3D asset generation.

## 🌍 Key Features

- **AI-Powered Analysis**: NVIDIA Nemotron VLM for intelligent product understanding
- **Smart Categorization**: Automatic classification into predefined product categories
- **Intelligent Prompt Planning**: Context-aware image variation planning based on regional aesthetics
- **Multi-Language Support**: Generate product titles and descriptions in **10 regional locales**
- **Cultural Image Generation**: Create culturally-appropriate product backgrounds (Spanish courtyards, Mexican family spaces, British formal settings)
- **3D Asset Generation**: Transform 2D product images into interactive 3D GLB models using Microsoft TRELLIS

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
- `POST /generate/3d` → **3D asset generation with TRELLIS** (~30-120 seconds) - generate 3D GLB models from 2D images

### API Endpoints

#### Modular Pipeline Workflow

The API provides a modular approach for optimal performance and flexibility:

**1) Fast VLM Analysis (POST `/vlm/analyze`)** - Get product fields quickly (~2-5 seconds)
**2) Image Generation (POST `/generate/variation`)** - Generate 2D variations on demand (~30-60 seconds)
**3) 3D Asset Generation (POST `/generate/3d`)** - Generate 3D models on demand (~30-120 seconds)

Benefits of this approach:
- Display product information immediately to users
- Generate images and 3D assets in the background or on-demand
- Cache VLM results and generate multiple variations
- Better error handling for each step
- Parallel generation of multiple asset types

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

##### With Brand-Specific Instructions:
```bash
curl -X POST \
  -F "image=@bag.jpg;type=image/jpeg" \
  -F 'product_data={"title":"Beauty Product","description":"Nice cream"}' \
  -F "locale=en-US" \
  -F 'brand_instructions=Write the catalog as a professional expert in Sephora Beauty. Strictly use this tone and style when writing the product document. Use this example as guidance for fragrance products: Title: Good Girl Blush Eau de Parfum with Floral Vanilla Description: A fresh, floral explosion of femininity, this radiant reinvention of the iconic Good Girl scent reveals the multifaceted nature of modern womanhood with a double dose of sensual vanilla and exotic ylang-ylang.' \
  http://localhost:8000/vlm/analyze
```

**Request Parameters:**
- `image` (required): Product image file
- `locale` (optional, default: "en-US"): Regional locale code for language/terminology
- `product_data` (optional): Existing product data JSON to augment
- `brand_instructions` (optional): Custom brand voice, tone, style, and taxonomy guidelines

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

#### 3️⃣ 3D Asset Generation: `/generate/3d`

Generate interactive 3D GLB models from 2D product images using Microsoft's TRELLIS model.

**Setup:** Configure TRELLIS endpoint in `shared/config/config.yaml`:
```yaml
trellis:
  url: "http://localhost:8006/v1/infer"
```

**Usage Examples:**

##### Basic Usage (Binary GLB Response):
```bash
curl -X POST \
  -F "image=@bag.jpg;type=image/jpeg" \
  http://localhost:8000/generate/3d \
  --output product.glb
```

##### With Custom Parameters:
```bash
curl -X POST \
  -F "image=@bag.jpg;type=image/jpeg" \
  -F "slat_cfg_scale=5.0" \
  -F "ss_cfg_scale=10.0" \
  -F "slat_sampling_steps=50" \
  -F "ss_sampling_steps=50" \
  -F "seed=42" \
  http://localhost:8000/generate/3d \
  --output product.glb
```

##### JSON Response (for Web Clients):
```bash
curl -X POST \
  -F "image=@bag.jpg;type=image/jpeg" \
  -F "return_json=true" \
  http://localhost:8000/generate/3d
```

**Request Parameters:**
- `image` (required): Product image file (JPEG, PNG)
- `slat_cfg_scale` (optional, default: 5.0): SLAT configuration scale
- `ss_cfg_scale` (optional, default: 10.0): SS configuration scale
- `slat_sampling_steps` (optional, default: 50): SLAT sampling steps
- `ss_sampling_steps` (optional, default: 50): SS sampling steps
- `seed` (optional, default: 0): Random seed for reproducibility
- `return_json` (optional, default: false): Return JSON with base64 GLB instead of binary

**Response (Binary Mode):**
- Binary GLB file (model/gltf-binary) ready for download

**Response (JSON Mode):**
```json
{
  "glb_base64": "Z2xURgIAAAA...",
  "artifact_id": "trellis_42",
  "metadata": {
    "slat_cfg_scale": 5.0,
    "ss_cfg_scale": 10.0,
    "slat_sampling_steps": 50,
    "ss_sampling_steps": 50,
    "seed": 42,
    "size_bytes": 1234567
  }
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


