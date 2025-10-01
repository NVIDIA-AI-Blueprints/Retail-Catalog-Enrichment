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
- GET `/` → plaintext greeting
- GET `/health` → `{ "status": "ok" }`
- POST `/vlm/describe`
  - Request: `multipart/form-data` with fields:
    - `image` (file): Product image
    - `product_data` (JSON string): Existing product data to augment
    - `locale` (string, optional): Regional locale code (default: "en-US")
  - Response: Enhanced product JSON with:
    - `title`: string (enriched and localized)
    - `description`: string (expanded and localized)
    - `categories`: array (validated/improved, e.g. ["accessories", "bag"])
    - `tags`: array (expanded with relevant terms)
    - `colors`: array (extracted color palette, e.g. ["black", "gold"])


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
- Preserves existing structured data (price, specs, etc.) while enhancing descriptive fields

#### Examples
```bash
# Image only (generation mode) - Default American English
curl -X POST \
  -F "image=@bag.jpg;type=image/jpeg" \
  -F "locale=en-US" \
  http://localhost:8000/vlm/describe

# With product data (augmentation mode) - American English
curl -X POST \
  -F "image=@bag.jpg;type=image/jpeg" \
  -F 'product_data={"title":"Classic Black Patent purse","description":"Elegant bag","price":15.99,"categories":["accessories","bag"],"tags":["bag","purse"]}' \
  -F "locale=en-US" \
  http://localhost:8000/vlm/describe

# Spain Spanish with product data
curl -X POST \
  -F "image=@bag.jpg;type=image/jpeg" \
  -F 'product_data={"categories":["accessories"],"title":"Black Purse","description":"Elegant bag"}' \
  -F "locale=es-ES" \
  http://localhost:8000/vlm/describe

# British English with full product data
curl -X POST \
  -F "image=@chair.jpg;type=image/jpeg" \
  -F 'product_data={"categories":["furniture"],"title":"Office Chair","description":"Comfortable chair","price":199.99}' \
  -F "locale=en-GB" \
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
