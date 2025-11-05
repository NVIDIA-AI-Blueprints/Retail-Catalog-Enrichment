# API Documentation

This document provides detailed information about the Catalog Enrichment System API endpoints.

## Base URL

- **Local Development**: `http://localhost:8000`
- **Docker Deployment**: `http://localhost:8000`

## Health & Info Endpoints

### GET `/`
Returns a plaintext greeting message.

**Response**: 
```
Welcome to Catalog Enrichment API
```

### GET `/health`
Health check endpoint for monitoring service status.

**Response**:
```json
{
  "status": "ok"
}
```

---

## API Endpoints

### Modular Pipeline Workflow

The API provides a modular approach for optimal performance and flexibility:

**1) Fast VLM Analysis (POST `/vlm/analyze`)** - Get product fields quickly
**2) Image Generation (POST `/generate/variation`)** - Generate 2D variations on demand  
**3) 3D Asset Generation (POST `/generate/3d`)** - Generate 3D models on demand

**Benefits of this approach:**
- Display product information immediately to users
- Generate images and 3D assets in the background or on-demand
- Cache VLM results and generate multiple variations
- Better error handling for each step
- Parallel generation of multiple asset types

---

## 1️⃣ Fast VLM Analysis: `/vlm/analyze`

Extract product fields using NVIDIA Nemotron VLM (no image generation).

**Endpoint**: `POST /vlm/analyze`  
**Content-Type**: `multipart/form-data`

### Request Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `image` | file | Yes | Product image file (JPEG, PNG) |
| `locale` | string | No | Regional locale code (default: "en-US") |
| `product_data` | JSON string | No | Existing product data to augment |
| `brand_instructions` | string | No | Custom brand voice, tone, style, and taxonomy guidelines |

### Product Data Schema (Optional)

```json
{
  "title": "string",
  "description": "string",
  "price": "number",
  "categories": ["string"],
  "tags": ["string"]
}
```

### Response Schema

```json
{
  "title": "string",
  "description": "string",
  "categories": ["string"],
  "tags": ["string"],
  "colors": ["string"],
  "locale": "string"
}
```

### Usage Examples

#### Image Only (Generation Mode)
```bash
curl -X POST \
  -F "image=@bag.jpg;type=image/jpeg" \
  -F "locale=en-US" \
  http://localhost:8000/vlm/analyze
```

#### With Existing Product Data (Augmentation Mode)
```bash
curl -X POST \
  -F "image=@bag.jpg;type=image/jpeg" \
  -F 'product_data={"title":"Classic Black Patent purse","description":"Elegant bag","price":15.99,"categories":["accessories"],"tags":["bag","purse"]}' \
  -F "locale=en-US" \
  http://localhost:8000/vlm/analyze
```

#### Regional Localization (Spain Spanish)
```bash
curl -X POST \
  -F "image=@bag.jpg;type=image/jpeg" \
  -F 'product_data={"title":"Black Purse","description":"Elegant bag"}' \
  -F "locale=es-ES" \
  http://localhost:8000/vlm/analyze
```

#### With Brand-Specific Instructions
```bash
curl -X POST \
  -F "image=@bag.jpg;type=image/jpeg" \
  -F 'product_data={"title":"Beauty Product","description":"Nice cream"}' \
  -F "locale=en-US" \
  -F 'brand_instructions=Write the catalog as a professional expert in Sephora Beauty. Strictly use this tone and style when writing the product document. Use this example as guidance for fragrance products: Title: Good Girl Blush Eau de Parfum with Floral Vanilla Description: A fresh, floral explosion of femininity, this radiant reinvention of the iconic Good Girl scent reveals the multifaceted nature of modern womanhood with a double dose of sensual vanilla and exotic ylang-ylang.' \
  http://localhost:8000/vlm/analyze
```

### Example Response

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

## 2️⃣ Image Generation: `/generate/variation`

Generate culturally-appropriate product variations using FLUX models based on VLM analysis results.

**Endpoint**: `POST /generate/variation`  
**Content-Type**: `multipart/form-data`

### Request Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `image` | file | Yes | Product image file (JPEG, PNG) |
| `title` | string | Yes | Product title from VLM analysis |
| `description` | string | Yes | Product description from VLM analysis |
| `categories` | JSON string | Yes | Categories array from VLM analysis |
| `locale` | string | No | Regional locale code (default: "en-US") |
| `tags` | JSON string | No | Tags array from VLM analysis |
| `colors` | JSON string | No | Colors array from VLM analysis |
| `enhanced_product` | JSON string | No | Enhanced product data |

### Response Schema

```json
{
  "generated_image_b64": "string (base64)",
  "artifact_id": "string",
  "image_path": "string",
  "metadata_path": "string",
  "locale": "string"
}
```

### Usage Example

```bash
# First, run VLM analysis to get the fields, then:
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

### Example Response

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

## 3️⃣ 3D Asset Generation: `/generate/3d`

Generate interactive 3D GLB models from 2D product images using Microsoft's TRELLIS model.

**Endpoint**: `POST /generate/3d`  
**Content-Type**: `multipart/form-data`

### Request Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `image` | file | Yes | - | Product image file (JPEG, PNG) |
| `slat_cfg_scale` | float | No | 5.0 | SLAT configuration scale |
| `ss_cfg_scale` | float | No | 10.0 | SS configuration scale |
| `slat_sampling_steps` | int | No | 50 | SLAT sampling steps |
| `ss_sampling_steps` | int | No | 50 | SS sampling steps |
| `seed` | int | No | 0 | Random seed for reproducibility |
| `return_json` | bool | No | false | Return JSON with base64 GLB or binary GLB |

### Response Formats

#### Binary Mode (default)
Returns binary GLB file (`model/gltf-binary`) ready for download.

#### JSON Mode
```json
{
  "glb_base64": "string (base64)",
  "artifact_id": "string",
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

### Usage Examples

#### Basic Usage (Binary GLB Response)
```bash
curl -X POST \
  -F "image=@bag.jpg;type=image/jpeg" \
  http://localhost:8000/generate/3d \
  --output product.glb
```

#### With Custom Parameters
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

#### JSON Response (for Web Clients)
```bash
curl -X POST \
  -F "image=@bag.jpg;type=image/jpeg" \
  -F "return_json=true" \
  http://localhost:8000/generate/3d
```

---

## Supported Locales

The API supports 10 regional locales for language and cultural context:

### English Variants
- `en-US` - American English (default)
- `en-GB` - British English  
- `en-AU` - Australian English
- `en-CA` - Canadian English

### Spanish Variants
- `es-ES` - Spain Spanish (uses "ordenador")
- `es-MX` - Mexican Spanish (uses "computadora") 
- `es-AR` - Argentinian Spanish
- `es-CO` - Colombian Spanish

### French Variants
- `fr-FR` - Metropolitan French
- `fr-CA` - Quebec French (Canadian)

---

## Error Responses

All endpoints return standard HTTP status codes:

- **200**: Success
- **400**: Bad Request (invalid parameters)
- **422**: Unprocessable Entity (validation error)
- **500**: Internal Server Error

Error response format:
```json
{
  "detail": "Error message description"
}
```
