# AGENTS.md - AI Assistant Instructions

This document provides guidelines and instructions for AI assistants working on the catalog-enrichment project.

## Project Overview

**Project Name:** catalog-enrichment  
**Repository:** https://gitlab-master.nvidia.com/anmartinez/catalog-enrichment.git  
**Purpose:** A system for enriching product catalog data with additional metadata, descriptions, and categorization.

### Current Status
- ✅ **Multi-Language Support** - Locale-based product descriptions (FR-6 completed)
- ✅ **VLM Content Augmentation** - Enhances existing product data with visual insights (FR-2 completed)
- ✅ **2D Image Variation Generation** - Working with prompt planning (FR-3 completed)
- ⚠️ **In Development** - 3D Asset Generation and Video Generation in progress

### Key Goals
- Enhance product catalog data with AI-powered enrichment
- Implement scalable data processing pipelines
- Ensure data quality and consistency
- Provide APIs for catalog data access and manipulation

## Build and Test Commands

### Prerequisites
Since this is an early-stage project, specific dependencies will be documented as they are added.

### Common Commands
```bash
# Clone the repository
git clone https://gitlab-master.nvidia.com/anmartinez/catalog-enrichment.git
cd catalog-enrichment

```

### Backend (current)

- Stack: FastAPI + Uvicorn (ASGI), OpenAI client (NVIDIA endpoint), Starlette under the hood
- Dependencies: `fastapi`, `uvicorn[standard]`, `openai`, `python-multipart`, `python-dotenv`, `langgraph==0.6.7`
- Python: 3.11+

#### Environment
- Create `.env` at repo root:
  - `NVIDIA_API_KEY=...`
  - `NVIDIA_API_BASE_URL=https://integrate.api.nvidia.com/v1` (default)

#### Run (with uv)
```bash
uv pip install -e .
uv venv .venv
source .venv/bin/activate
uvicorn --app-dir src backend.main:app --host 0.0.0.0 --port 8000 --reload
```

#### API Endpoints

**Health & Info:**
- GET `/` → plaintext greeting
- GET `/health` → `{ "status": "ok" }`

**VLM Analysis (Fast - ~2-5 seconds):**
- POST `/vlm/analyze`
  - Request: `multipart/form-data` with fields:
    - `image` (file): Product image
    - `product_data` (JSON string, optional): Existing product data to augment
    - `locale` (string, optional): Regional locale code (default: "en-US")
    - `brand_instructions` (string, optional): Custom brand voice, tone, style, and taxonomy guidelines
  - Response: Product fields JSON with:
    - `title`: string (enriched and localized)
    - `description`: string (expanded and localized)
    - `categories`: array (validated/improved, e.g. ["accessories", "bag"])
    - `tags`: array (expanded with relevant terms)
    - `colors`: array (extracted color palette, e.g. ["black", "gold"])
    - `enhanced_product`: object (if product_data was provided)

**Image Generation (Slow - ~30-60 seconds):**
- POST `/generate/variation`
  - Request: `multipart/form-data` with fields:
    - `image` (file): Product image
    - `title` (string): Product title from VLM analysis
    - `description` (string): Product description from VLM analysis
    - `categories` (JSON string): Categories array from VLM analysis
    - `tags` (JSON string, optional): Tags array from VLM analysis
    - `colors` (JSON string, optional): Colors array from VLM analysis
    - `locale` (string, optional): Regional locale code (default: "en-US")
    - `enhanced_product` (JSON string, optional): Enhanced product data
  - Response: Generated image JSON with:
    - `generated_image_b64`: string (base64-encoded PNG)
    - `artifact_id`: string (unique identifier)
    - `image_path`: string (disk location of saved image)
    - `metadata_path`: string (disk location of saved metadata)

**3D Asset Generation (Slow - ~30-120 seconds):**
- POST `/generate/3d`
  - Request: `multipart/form-data` with fields:
    - `image` (file): Product image (JPEG, PNG)
    - `slat_cfg_scale` (float, optional): SLAT configuration scale (default: 5.0)
    - `ss_cfg_scale` (float, optional): SS configuration scale (default: 10.0)
    - `slat_sampling_steps` (int, optional): SLAT sampling steps (default: 50)
    - `ss_sampling_steps` (int, optional): SS sampling steps (default: 50)
    - `seed` (int, optional): Random seed for reproducibility (default: 0)
    - `return_json` (bool, optional): Return JSON with base64 GLB or binary GLB (default: false)
  - Response: Binary GLB file (model/gltf-binary) or JSON with:
    - `glb_base64`: string (base64-encoded GLB file)
    - `artifact_id`: string (unique identifier)
    - `metadata`: object (generation parameters and file size)

**Legacy (Complete Pipeline - ~35-65 seconds):**
- POST `/vlm/describe`
  - Request: `multipart/form-data` with fields:
    - `image` (file): Product image
    - `product_data` (JSON string, optional): Existing product data to augment
    - `locale` (string, optional): Regional locale code (default: "en-US")
  - Response: Complete JSON with fields + generated image
    - All fields from `/vlm/analyze`
    - `generated_image_b64`: string (base64-encoded PNG)


Input Product Data Schema (optional):
```json
{
  "title": "string",
  "description": "string",
  "price": "number",
  "categories": ["string"],
  "tags": ["string"]
}
```

Example Response:
```json
{
  "title": "Glamorous Black Evening Handbag with Gold Accents",
  "description": "This exquisite handbag exudes sophistication and elegance...",
  "categories": ["accessories"],
  "tags": ["black leather", "gold accents", "evening bag", "rectangular shape"],
  "colors": ["black", "gold"]
}
```

Supported locales: `en-US`, `en-GB`, `en-AU`, `en-CA`, `es-ES`, `es-MX`, `es-AR`, `es-CO`, `fr-FR`, `fr-CA`

#### VLM Prompt (summary)
- Instructs the model to augment existing product data by analyzing the product image
- Enriches the `title` with more descriptive and persuasive language in the specified regional locale
- Expands the `description` with richer, more detailed content using regional terminology
- Enhances `attributes` with visual insights (e.g., "Black" → "Matte Black with Silver Hardware")
- Validates and improves `categories` array based on visual analysis
- Expands `tags` with additional relevant terms
- Includes regional context and terminology guidance (e.g., "ordenador" vs "computadora" for Spanish regions)
- Applies custom brand instructions when provided for consistent tone, voice, and taxonomy
- Preserves existing structured data (price, specs, etc.) while enhancing descriptive fields

#### Examples

**Fast VLM Analysis (returns fields in ~2-5 seconds):**
```bash
# Analyze image only - Default American English
curl -X POST \
  -F "image=@bag.jpg;type=image/jpeg" \
  -F "locale=en-US" \
  http://localhost:8000/vlm/analyze

# With product data (augmentation mode) - American English
curl -X POST \
  -F "image=@bag.jpg;type=image/jpeg" \
  -F 'product_data={"title":"Classic Black Patent purse","description":"Elegant bag","price":15.99,"categories":["accessories","bag"],"tags":["bag","purse"]}' \
  -F "locale=en-US" \
  http://localhost:8000/vlm/analyze

# Spain Spanish with product data
curl -X POST \
  -F "image=@bag.jpg;type=image/jpeg" \
  -F 'product_data={"categories":["accessories"],"title":"Black Purse","description":"Elegant bag"}' \
  -F "locale=es-ES" \
  http://localhost:8000/vlm/analyze

# With brand-specific instructions for custom tone and taxonomy
curl -X POST \
  -F "image=@product.jpg;type=image/jpeg" \
  -F 'product_data={"title":"Beauty Product","description":"Nice cream"}' \
  -F "locale=en-US" \
  -F 'brand_instructions=You work at a premium beauty retailer. Use a playful, empowering, and inclusive brand voice. Focus on self-expression and beauty discovery. Use terms like "beauty lovers", "glow", "radiant", and "treat yourself". Our product taxonomy emphasizes skin benefits and ingredient transparency.' \
  http://localhost:8000/vlm/analyze
```

**Image Generation (returns generated image in ~30-60 seconds):**
```bash
# Generate variation using VLM results
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

**3D Asset Generation (returns GLB file in ~30-120 seconds):**
```bash
# Generate 3D GLB asset with default parameters (returns binary GLB)
curl -X POST \
  -F "image=@bag.jpg;type=image/jpeg" \
  http://localhost:8000/generate/3d \
  --output product.glb

# Generate with custom parameters
curl -X POST \
  -F "image=@bag.jpg;type=image/jpeg" \
  -F "slat_cfg_scale=5.0" \
  -F "ss_cfg_scale=10.0" \
  -F "slat_sampling_steps=50" \
  -F "ss_sampling_steps=50" \
  -F "seed=42" \
  http://localhost:8000/generate/3d \
  --output product.glb

# Return JSON with base64-encoded GLB (useful for web clients)
curl -X POST \
  -F "image=@bag.jpg;type=image/jpeg" \
  -F "return_json=true" \
  http://localhost:8000/generate/3d
```

**Complete Pipeline (legacy endpoint - returns everything in ~35-65 seconds):**
```bash
# Image only (generation mode)
curl -X POST \
  -F "image=@bag.jpg;type=image/jpeg" \
  -F "locale=en-US" \
  http://localhost:8000/vlm/describe

# With product data (augmentation mode)
curl -X POST \
  -F "image=@bag.jpg;type=image/jpeg" \
  -F 'product_data={"title":"Classic Black Patent purse","description":"Elegant bag"}' \
  -F "locale=en-US" \
  http://localhost:8000/vlm/describe
```

**Note:** Build and test commands will be updated as the project's technology stack is finalized.

## Code Style Guidelines

### General Principles
- **Clarity over cleverness** - Write code that is easy to understand and maintain
- **Consistent formatting** - Use automated formatting tools when available
- **Meaningful names** - Use descriptive variable, function, and class names
- **Documentation** - Include docstrings and comments for complex logic

### File Organization
- Keep files focused on a single responsibility
- Use clear, descriptive file names
- Organize code into logical directories/modules
- Separate configuration from business logic

## Testing Instructions

### Testing Strategy
- **Unit Tests** - Test individual functions and components in isolation
- **Integration Tests** - Test interactions between system components
- **Data Validation Tests** - Ensure data integrity and format compliance
- **Performance Tests** - Validate system performance under load

### Test Organization
- Place tests in a dedicated `tests/` directory or alongside source code
- Mirror the source code structure in test organization
- Use descriptive test names that explain what is being tested

### Data Testing
Given the catalog enrichment focus, pay special attention to:
- Data validation and sanitization
- Schema compliance
- Data transformation accuracy
- Edge cases and malformed input handling

### Coverage Goals
- Aim for >80% code coverage on critical paths
- Prioritize testing of data processing and enrichment logic
- Include error handling and edge case scenarios

## Security Considerations

### Data Security
- **PII Protection** - Identify and protect personally identifiable information
- **Data Classification** - Understand and respect data sensitivity levels
- **Access Controls** - Implement appropriate authentication and authorization
- **Data Encryption** - Encrypt sensitive data in transit and at rest

### API Security
- Validate all input data
- Implement rate limiting to prevent abuse
- Use HTTPS for all external communications
- Log security events for monitoring

### Development Security
- Keep dependencies up to date
- Use environment variables for sensitive configuration
- Never commit credentials or secrets to version control
- Conduct security reviews for critical changes

### NVIDIA-Specific Considerations
- Follow NVIDIA security policies and guidelines
- Ensure compliance with enterprise security requirements
- Use approved tools and services where applicable

## AI Assistant Guidelines

### When Working on This Project

1. **Understand the Context**
   - This is a data-centric project focused on catalog enrichment
   - Consider data quality, performance, and scalability in all decisions
   - Be mindful of the enterprise environment (NVIDIA internal)

2. **Code Quality**
   - Always run tests before suggesting changes
   - Ensure new code follows established patterns
   - Include appropriate error handling and logging

3. **Documentation**
   - Update relevant documentation when making changes
   - Include examples in API documentation
   - Keep this AGENTS.md file current as the project evolves

4. **Communication**
   - Ask for clarification when requirements are ambiguous
   - Suggest improvements to architecture and processes
   - Flag potential security or performance concerns

5. **Incremental Development**
   - Start with simple, working solutions
   - Iterate and improve based on feedback
   - Consider backwards compatibility when making changes

---

**Last Updated:** $(date)  
**Version:** 1.0  

*This document should be updated as the project evolves and new practices are established.*
