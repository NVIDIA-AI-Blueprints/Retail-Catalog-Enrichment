"""
Image Generation Pipeline

Handles the image variation generation workflow:
Planner → FLUX → Persist

This module is decoupled from VLM analysis and takes pre-computed fields as input.
"""
import os
import json
import base64
import random
import logging
import httpx
from uuid import uuid4
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, TypedDict
from io import BytesIO
from PIL import Image

from dotenv import load_dotenv
from openai import OpenAI
from backend.config import get_config

load_dotenv()

logger = logging.getLogger("catalog_enrichment.image")

# Locale configuration for planner
LOCALE_CONFIG = {
    "en-US": {"language": "English", "region": "United States", "country": "United States", "context": "American English with US terminology (e.g., 'cell phone', 'sweater')"},
    "en-GB": {"language": "English", "region": "United Kingdom", "country": "United Kingdom", "context": "British English with UK terminology (e.g., 'mobile phone', 'jumper')"},
    "en-AU": {"language": "English", "region": "Australia", "country": "Australia", "context": "Australian English with local terminology"},
    "en-CA": {"language": "English", "region": "Canada", "country": "Canada", "context": "Canadian English"},
    "es-ES": {"language": "Spanish", "region": "Spain", "country": "Spain", "context": "Peninsular Spanish with Spain-specific terminology (e.g., 'ordenador' for computer)"},
    "es-MX": {"language": "Spanish", "region": "Mexico", "country": "Mexico", "context": "Mexican Spanish with Latin American terminology (e.g., 'computadora' for computer)"},
    "es-AR": {"language": "Spanish", "region": "Argentina", "country": "Argentina", "context": "Argentinian Spanish with local expressions"},
    "es-CO": {"language": "Spanish", "region": "Colombia", "country": "Colombia", "context": "Colombian Spanish"},
    "fr-FR": {"language": "French", "region": "France", "country": "France", "context": "Metropolitan French"},
    "fr-CA": {"language": "French", "region": "Canada", "country": "Canada", "context": "Quebec French with Canadian terminology"}
}


# VLMState type (needed for node functions)
class VLMState(TypedDict, total=False):
    image_bytes: bytes
    content_type: str
    locale: str
    product_data: Optional[Dict[str, Any]]
    title: str
    description: str
    categories: List[str]
    tags: List[str]
    colors: List[str]
    error: Optional[str]
    generated_image_b64: str
    image_path: str
    metadata_path: str
    artifact_id: str
    variation_plan: Dict[str, Any]
    flux_prompt: str
    enhanced_product: Dict[str, Any]


def _call_planner_llm(title: str, description: str, categories: List[str], locale: str = "en-US") -> Dict[str, Any]:
    """Call the planner LLM to create an image variation plan."""
    logger.info("Calling planner LLM: title_len=%d desc_len=%d cats=%s locale=%s", len(title or ""), len(description or ""), categories, locale)

    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError("NVIDIA_API_KEY is not set")

    country = LOCALE_CONFIG.get(locale, {"country": "United States"})["country"]
    
    llm_config = get_config().get_llm_config()
    client = OpenAI(base_url=llm_config['url'], api_key=api_key)

    completion = client.chat.completions.create(
        model=llm_config['model'],
        messages=[
            {"role": "system", "content": "/no_think You are a product image variation planner with cultural awareness. Output ONLY valid JSON - no markdown formatting, no code blocks, no backticks. "
             "Preserve the subject identity. Change ONLY background, camera angle, lighting, and mood according to the title, description, and target locale. "
             "Create backgrounds that reflect the cultural aesthetic and lifestyle of the target region!"
             "Adhere to the JSON schema with fields: preserve_subject, background_style, camera_angle, lighting, color_palette, negatives, cfg_scale, steps, variants."},
            {"role": "user", "content": f"""TITLE: {title}
DESCRIPTION: {description}
CATEGORIES: {categories}
TARGET LOCALE: {locale}

Create a background style that authentically reflects how this product would be used in {country}. Use your knowledge of local architecture, interior design, lifestyle, and cultural preferences for that country.

BE CREATIVE AND VARY YOUR CHOICES:
- Consider diverse settings: home interiors, outdoor scenes, professional environments, lifestyle contexts
- Think beyond obvious choices - vary backgrounds based on product type and cultural context
- Use different camera angles: overhead, eye-level, low angle, 3/4 view, close-up
- Vary lighting: natural window light, golden hour, studio softbox, dramatic side light, diffused overcast
- Avoid repetitive patterns - each generation should feel unique

CATEGORY-SPECIFIC BACKGROUNDS:
- For "skincare" products: use bathroom counters, bathroom vanity setups, living room side tables, bedroom vanities, or spa-inspired settings
- For "kitchen" products: use kitchen counters, dining tables, or cooking prep areas
- For "accessories": use lifestyle contexts like entryways, closets, or fashion displays
- For other categories: choose contextually appropriate settings that match how the product is typically used

Produce ONLY a JSON object with no markdown formatting or code blocks. Required schema:
{{"preserve_subject": "<exact product description from title>", 
"background_style": "<culturally authentic setting for {country} - be specific and creative>", 
"camera_angle": "<varied: overhead/eye-level/low angle/3-4 view/close-up>", 
"lighting": "<varied: natural window/golden hour/studio softbox/side light/overcast/etc>", 
"color_palette": "<complement the product and setting>", 
"negatives": ["do not alter the subject", "no text, no logos, no duplicates"], 
"cfg_scale": <float between 2.5-4.5>, "steps": <int 25-40>, "variants": 1}}

CRITICAL: Return the raw JSON object only - no ```json``` or ``` blocks. Keep the subject unchanged. Do not add extra keys or commentary. Make each background contextually appropriate AND visually distinct."""}
        ],
        temperature=0.8, top_p=1, max_tokens=1024, stream=True
    )

    text = "".join(chunk.choices[0].delta.content for chunk in completion if chunk.choices[0].delta and chunk.choices[0].delta.content).strip()
    logger.info("Planner LLM response received: %s", text)
    
    json_text = text
    if "```json" in text:
        try:
            start = text.find("```json") + len("```json")
            end = text.find("```", start)
            if end > start:
                json_text = text[start:end].strip()
                logger.info("Extracted JSON from markdown: %s", json_text)
        except Exception as e:
            logger.warning("Failed to extract JSON from markdown: %s", e)
    elif "```" in text:
        try:
            start = text.find("```") + len("```")
            end = text.find("```", start)
            if end > start:
                json_text = text[start:end].strip()
                logger.info("Extracted JSON from generic code block: %s", json_text)
        except Exception as e:
            logger.warning("Failed to extract JSON from code block: %s", e)
    
    try:
        plan = json.loads(json_text)
        if isinstance(plan, dict):
            logger.info("Successfully parsed planner JSON with keys: %s", list(plan.keys()))
            return plan
    except Exception as e:
        logger.warning("Planner LLM returned non-JSON; using fallback plan. Parse error: %s", e)
    
    # Randomized fallback options for variety
    backgrounds = [
        "neutral studio cyclorama", 
        "minimalist table surface, shallow depth of field",
        "natural wooden surface with soft shadows",
        "lifestyle setting with blurred background",
        "clean white surface, high key lighting"
    ]
    camera_angles = ["eye-level", "overhead", "3/4 view", "slight angle"]
    lighting_options = ["softbox, high key", "natural window light", "diffused studio lighting", "soft directional light"]
    
    return {
        "preserve_subject": title or "product",
        "background_style": random.choice(backgrounds),
        "camera_angle": random.choice(camera_angles),
        "lighting": random.choice(lighting_options),
        "color_palette": "neutral",
        "negatives": ["do not alter the subject", "no text, no logos, no duplicates"],
        "cfg_scale": round(random.uniform(2.8, 4.0), 1), 
        "steps": random.choice([28, 30, 32, 35]), 
        "variants": 1
    }


def _render_flux_prompt(plan: Dict[str, Any]) -> str:
    """Render a FLUX prompt from a variation plan."""
    preserve = plan.get("preserve_subject", "the product")
    background = plan.get("background_style", "neutral studio background")
    camera = plan.get("camera_angle", "eye-level")
    lighting = plan.get("lighting", "softbox")
    negatives = plan.get("negatives", [])
    neg_text = "; ".join(negatives) if isinstance(negatives, list) else str(negatives)
    
    return f"Keep {preserve} unchanged. Replace only the background with {background}. Make it hyperrealistic, ideal for an e-commerce product image. " \
           f"Use {lighting} lighting and {camera} camera angle. " \
           f"Maintain subject color, orientation, and material. Scale the product to natural, proportional size for the environment. " \
           f"{('Avoid: ' + neg_text) if neg_text else ''}".strip()


async def _call_flux_edit(image_bytes: bytes, content_type: str, prompt: str, steps: int, cfg_scale: float, seed: Optional[int] = None) -> Dict[str, Any]:
    """Call the FLUX image editing API."""
    logger.info("Calling FLUX edit: prompt_len=%d steps=%d cfg=%.2f", len(prompt), steps, cfg_scale)
    
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError("NVIDIA_API_KEY is not set")

    flux_config = get_config().get_flux_config()

    try:
        image = Image.open(BytesIO(image_bytes))
        png_buffer = BytesIO()
        image.save(png_buffer, format='PNG')
        png_bytes = png_buffer.getvalue()
        data_url = f"data:image/png;base64,{base64.b64encode(png_bytes).decode()}"
        logger.info("Image converted to PNG for FLUX: size=%d bytes", len(png_bytes))
    except Exception as e:
        logger.warning("PNG conversion failed, using original: %s", e)
        data_url = f"data:{content_type};base64,{base64.b64encode(image_bytes).decode()}"
    
    logger.info("FLUX prompt: %s", prompt)

    async with httpx.AsyncClient(timeout=180.0) as client:
        response = await client.post(
            flux_config['url'],
            headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json", "Content-Type": "application/json"},
            json={"prompt": prompt, "image": data_url, "aspect_ratio": "match_input_image", 
                  "steps": int(steps or 30), "cfg_scale": float(cfg_scale or 3.5), "seed": int(seed if seed is not None else 0)}
        )
    
    body = response.json()
    logger.info("FLUX response received: keys=%s", list(body.keys()))
    return body


def _extract_base64_image_from_flux_response(response_body: Dict[str, Any]) -> Optional[str]:
    """Extract base64 image from FLUX API response."""
    for key in ("image", "output", "data"):
        val = response_body.get(key)
        if isinstance(val, str) and val:
            return val
    
    for collection_key in ("images", "artifacts"):
        collection = response_body.get(collection_key)
        if isinstance(collection, list) and collection:
            first = collection[0]
            if isinstance(first, str) and first:
                return first
            if isinstance(first, dict):
                for key in ("b64", "base64", "image"):
                    val = first.get(key)
                    if isinstance(val, str) and val:
                        return val
    return None


def planner_node(state: VLMState) -> VLMState:
    """LangGraph node: Create variation plan from VLM analysis."""
    logger.info("Planner node start")
    try:
        locale = state.get("locale", "en-US")
        plan = _call_planner_llm(state.get("title", "").strip(), state.get("description", "").strip(), state.get("categories", []), locale)
        prompt = _render_flux_prompt(plan)
        logger.info("Planner node success: prompt_len=%d locale=%s", len(prompt), locale)
        return {**state, "variation_plan": plan, "flux_prompt": prompt}
    except Exception as exc:
        logger.exception("Planner node exception: %s", exc)
        return {"error": str(exc), **state}


async def flux_node(state: VLMState) -> VLMState:
    """LangGraph node: Generate image variation using FLUX."""
    logger.info("FLUX node start")
    image_bytes = state.get("image_bytes")
    prompt = (state.get("flux_prompt") or "").strip()

    if not image_bytes or not prompt:
        error = "image_bytes missing" if not image_bytes else "flux_prompt missing"
        logger.error(f"FLUX node error: {error}")
        return {"error": error, **state}

    try:
        plan = state.get("variation_plan", {})
        flux_response = await _call_flux_edit(
            image_bytes, state.get("content_type", "image/png"), prompt,
            int(plan.get("steps", 30)), float(plan.get("cfg_scale", 3.5)),
            random.randint(1, 10_000_000)
        )
        image_b64 = _extract_base64_image_from_flux_response(flux_response)
        if not image_b64:
            return {"error": "FLUX response did not include an image", **state}
        logger.info("FLUX node success: image_b64_len=%d", len(image_b64))
        return {**state, "generated_image_b64": image_b64}
    except Exception as exc:
        logger.exception("FLUX node exception: %s", exc)
        return {"error": str(exc), **state}


def persist_node(state: VLMState) -> VLMState:
    """LangGraph node: Persist generated image and metadata to disk."""
    logger.info("Persist node start")
    if not (image_b64 := state.get("generated_image_b64")):
        return {"error": "generated_image_b64 missing", **state}

    try:
        output_dir = os.getenv("OUTPUT_DIR", os.path.join(os.getcwd(), "data", "outputs"))
        os.makedirs(output_dir, exist_ok=True)

        artifact_id = uuid4().hex
        image_path = os.path.join(output_dir, f"{artifact_id}.png")
        metadata_path = os.path.join(output_dir, f"{artifact_id}.json")

        with open(image_path, "wb") as f:
            f.write(base64.b64decode(image_b64))

        base_meta = {
            "id": artifact_id,
            "locale": state.get("locale", "en-US"),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "image_path": image_path,
            "source_content_type": state.get("content_type")
        }
        
        if enhanced := state.get("enhanced_product"):
            logger.info("Persist node: saving enhanced_product with keys=%s", list(enhanced.keys()))
            metadata = {**enhanced, **base_meta}
        else:
            logger.info("Persist node: saving VLM-only data")
            metadata = {
                **base_meta,
                "title": state.get("title", ""),
                "description": state.get("description", ""),
                "categories": state.get("categories", []),
                "tags": state.get("tags", []),
                "colors": state.get("colors", [])
            }

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        logger.info("Persist node success: image_path=%s metadata_path=%s", image_path, metadata_path)
        return {**state, "artifact_id": artifact_id, "image_path": image_path, "metadata_path": metadata_path}
    except Exception as exc:
        logger.exception("Persist node exception: %s", exc)
        return {"error": str(exc), **state}


def persist_generated_image(
    image_b64: str,
    title: str,
    description: str,
    categories: List[str],
    tags: List[str],
    colors: List[str],
    locale: str,
    content_type: str,
    enhanced_product: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Persist generated image and metadata to disk.
    
    Returns artifact information (id, paths).
    """
    logger.info("Persisting generated image: title_len=%d locale=%s", len(title), locale)
    
    output_dir = os.getenv("OUTPUT_DIR", os.path.join(os.getcwd(), "data", "outputs"))
    os.makedirs(output_dir, exist_ok=True)
    
    artifact_id = uuid4().hex
    image_path = os.path.join(output_dir, f"{artifact_id}.png")
    metadata_path = os.path.join(output_dir, f"{artifact_id}.json")
    
    # Save image
    with open(image_path, "wb") as f:
        f.write(base64.b64decode(image_b64))
    
    # Prepare metadata
    base_meta = {
        "id": artifact_id,
        "locale": locale,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "image_path": image_path,
        "source_content_type": content_type
    }
    
    if enhanced_product:
        logger.info("Saving enhanced_product metadata with keys=%s", list(enhanced_product.keys()))
        metadata = {**enhanced_product, **base_meta}
    else:
        logger.info("Saving standard metadata")
        metadata = {
            **base_meta,
            "title": title,
            "description": description,
            "categories": categories,
            "tags": tags,
            "colors": colors
        }
    
    # Save metadata
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    logger.info("Persist success: artifact_id=%s image_path=%s", artifact_id, image_path)
    return {
        "artifact_id": artifact_id,
        "image_path": image_path,
        "metadata_path": metadata_path
    }


async def generate_image_variation(
    image_bytes: bytes,
    content_type: str,
    title: str,
    description: str,
    categories: List[str],
    tags: List[str],
    colors: List[str],
    locale: str = "en-US",
    enhanced_product: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Generate image variation given pre-computed VLM analysis results.
    
    Pipeline: Planner → FLUX → Persist
    
    Args:
        image_bytes: Original product image bytes
        content_type: Image MIME type
        title: Product title (from VLM)
        description: Product description (from VLM)
        categories: Product categories (from VLM)
        tags: Product tags (from VLM)
        colors: Product colors (from VLM)
        locale: Target locale for variation
        enhanced_product: Optional enhanced product data (if augmentation mode)
    
    Returns:
        Dict with generated_image_b64, artifact_id, image_path, metadata_path
    """
    logger.info("Starting image generation pipeline: title_len=%d locale=%s", len(title), locale)
    
    try:
        # Step 1: Planner - Create variation plan
        logger.info("Step 1: Planning variation")
        plan = _call_planner_llm(title, description, categories, locale)
        prompt = _render_flux_prompt(plan)
        logger.info("Planner complete: prompt_len=%d", len(prompt))
        
        # Step 2: FLUX - Generate image (async!)
        logger.info("Step 2: Generating image with FLUX")
        flux_response = await _call_flux_edit(
            image_bytes,
            content_type,
            prompt,
            steps=int(plan.get("steps", 30)),
            cfg_scale=float(plan.get("cfg_scale", 3.5)),
            seed=random.randint(1, 10_000_000)
        )
        
        image_b64 = _extract_base64_image_from_flux_response(flux_response)
        if not image_b64:
            raise RuntimeError("FLUX response did not include an image")
        
        logger.info("FLUX complete: image_b64_len=%d", len(image_b64))
        
        # Step 3: Persist - Save to disk
        logger.info("Step 3: Persisting artifact")
        artifact = persist_generated_image(
            image_b64=image_b64,
            title=title,
            description=description,
            categories=categories,
            tags=tags,
            colors=colors,
            locale=locale,
            content_type=content_type,
            enhanced_product=enhanced_product
        )
        
        logger.info("Image generation pipeline complete: artifact_id=%s", artifact["artifact_id"])
        
        return {
            "generated_image_b64": image_b64,
            "artifact_id": artifact["artifact_id"],
            "image_path": artifact["image_path"],
            "metadata_path": artifact["metadata_path"],
            "variation_plan": plan
        }
        
    except Exception as exc:
        logger.exception("Image generation pipeline failed: %s", exc)
        raise

