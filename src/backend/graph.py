import os
import json
import base64
import logging
from uuid import uuid4
import random
from datetime import datetime, timezone
from typing import TypedDict, Optional, List, Dict, Any
import requests
from io import BytesIO
from PIL import Image

from dotenv import load_dotenv
from openai import OpenAI
from langgraph.graph import StateGraph
from backend.config import get_config

load_dotenv()

logger = logging.getLogger("catalog_enrichment.graph")

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

# Allowed product categories for classification
PRODUCT_CATEGORIES = [
    "clothing",
    "kitchen", 
    "accessories",
    "toys",
    "electronics",
    "furniture",
    "office",
    "shoes"
]

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

def _call_nemotron_merge(vlm_output: Dict[str, Any], product_data: Dict[str, Any], locale: str = "en-US") -> Dict[str, Any]:
    """Intelligently merge VLM visual analysis with existing product data using Nemotron LLM."""
    logger.info("Calling Nemotron merge: vlm_keys=%s, product_keys=%s, locale=%s", 
                list(vlm_output.keys()), list(product_data.keys()), locale)
    
    if not (api_key := os.getenv("NVIDIA_API_KEY")):
        raise RuntimeError("NVIDIA_API_KEY is not set")

    info = LOCALE_CONFIG.get(locale, {"language": "English", "region": "United States", "country": "United States", "context": "American English"})
    llm_config = get_config().get_llm_config()
    client = OpenAI(base_url=llm_config['url'], api_key=api_key)

    vlm_json = json.dumps(vlm_output, indent=2, ensure_ascii=False)
    product_json = json.dumps(product_data, indent=2, ensure_ascii=False)

    prompt = f"""You are a product data merging specialist for an e-commerce platform. Your task is to intelligently merge visual analysis from a VLM with existing product data.

VISUAL ANALYSIS (from Vision Model - direct observation of the product image):
{vlm_json}

EXISTING PRODUCT DATA (may contain errors or be incomplete):
{product_json}

ALLOWED CATEGORIES (must use one or more from this list):
{json.dumps(PRODUCT_CATEGORIES)}

MERGING STRATEGY:

1. **Trust Priority**:
   - TRUST visual analysis for: colors, materials, design details, visual attributes
   - PRESERVE existing data for: price, SKU, specifications, dimensions (unless visually contradicted)
   - RESOLVE conflicts by prioritizing visual evidence when it contradicts existing data

2. **Title** (in {info['language']} for {info['region']}):
   - Intelligently merge both titles by:
     * Preserving brand terms, material descriptors, and unique identifiers from existing title
     * Incorporating visual details and attributes from VLM's title
     * Creating a natural, cohesive result that's richer than either source alone
   - If existing title is generic/poor and VLM title is comprehensive, prefer VLM
   - If existing title has valuable brand/material terms, integrate them with VLM insights
   - Apply regional terminology: {info['context']}

3. **Description** (in {info['language']} for {info['region']}):
   - Weave both sources into a unified, flowing narrative
   - Preserve brand voice and marketing language from existing description
   - Enrich with VLM's visual observations (materials, design details, features)
   - Avoid simple concatenation - integrate insights naturally
   - Use regional language and terminology

4. **Categories** (MUST be an array):
   - Validate existing categories against VLM observation
   - Use VLM's categories if existing is incorrect or ["uncategorized"]
   - MUST use categories from the allowed list above
   - Always return as an array: "categories": ["category1", "category2"]
   - Keep in English

5. **Attributes**:
   - Merge both dictionaries, with VLM's visual observations taking priority for conflicts
   - Example: existing "color": "Black" + VLM observes "Matte Black" → use "Matte Black"
   - Add new attributes from VLM if observed

6. **Tags**:
   - Combine tags from both sources, remove duplicates
   - Keep in English

7. **Specs & Price**:
   - Keep existing specs and price unchanged
   - Only add if VLM can visually verify (rare)

8. **Colors**:
   - Use VLM's color analysis (most accurate from visual observation)

OUTPUT FORMAT:
Return the merged data using the EXISTING PRODUCT DATA schema/structure. Preserve all original fields and add VLM insights where appropriate.

Return ONLY valid JSON with no markdown formatting or commentary."""

    completion = client.chat.completions.create(
        model=llm_config['model'],
        messages=[{"role": "system", "content": "/no_think"}, {"role": "user", "content": prompt}],
        temperature=0.5, top_p=0.9, max_tokens=2048, stream=True
    )

    text = "".join(chunk.choices[0].delta.content for chunk in completion if chunk.choices[0].delta and chunk.choices[0].delta.content)
    logger.info("Nemotron merge response received: %d chars", len(text))

    json_text = text.strip()
    for marker in ("```json", "```"):
        if marker in json_text:
            try:
                start = json_text.find(marker) + len(marker)
                end = json_text.find("```", start)
                if end > start:
                    json_text = json_text[start:end].strip()
                    break
            except Exception as e:
                logger.warning(f"Failed to extract JSON from {marker}: {e}")

    try:
        parsed = json.loads(json_text)
        if isinstance(parsed, dict):
            logger.info("Nemotron merge successful: merged_keys=%s", list(parsed.keys()))
            return parsed
    except Exception as e:
        logger.warning(f"Nemotron merge JSON parse error: {e}, using VLM output")
    
    return vlm_output

def _call_vlm(image_bytes: bytes, content_type: str, locale: str = "en-US") -> Dict[str, Any]:
    logger.info("Calling VLM: bytes=%d, content_type=%s, locale=%s", len(image_bytes or b""), content_type, locale)
    
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError("NVIDIA_API_KEY is not set")

    info = LOCALE_CONFIG.get(locale, {"language": "English", "region": "United States", "country": "United States", "context": "American English"})
    
    vlm_config = get_config().get_vlm_config()
    client = OpenAI(base_url=vlm_config['url'], api_key=api_key)

    categories_str = json.dumps(PRODUCT_CATEGORIES)
    
    prompt_text = f"""You are a product catalog copywriter for an e-commerce platform. Create compelling catalog content for the physical product shown in the image. Be verbose and detailed.

Focus on the actual product visible in the image - describe the item itself, its materials, design, and features. Do not focus on contents, intended use scenarios, or lifestyle experiences.

Create an engaging product title and description in {info['language']} as spoken in {info['region']}. {info['context']}.

Classify the product into one or more categories from this fixed allowed set only:
{categories_str}
If none apply, use ["uncategorized"].

Generate exactly 10 useful product tags that describe the item's characteristics, materials, style, features, or type. These should be short descriptive phrases (2-4 words each) that would help customers find this product. Use English for the tags.

Extract up to 3 primary colors visible in the product. Choose the most prominent and distinctive colors that would help customers identify or search for the product. Use simple, standard color names in English (e.g., "red", "blue", "black", "white", "brown", "grey", "green", "yellow", "orange", "purple", "pink", "navy", "beige", "cream", "silver", "gold"). If fewer than 3 distinct colors are clearly visible, provide only the colors you can confidently identify.

IMPORTANT GUIDELINES:
- Write compelling catalog copy that sells the physical product itself
- Highlight the product's materials, design, construction, and notable features
- Use persuasive but accurate language focused on the tangible item
- Avoid analytical observations like "appears to be" or "seems to be"
- Generate product-focused titles and descriptions using regional {info['language']} terminology
- Keep categories and tags in English as specified above
- Make tags specific and useful for search/filtering

Return ONLY valid JSON with the following structure:
{{
  "title": "<compelling product name describing the physical item in regional {info['language']}>",
  "description": "<persuasive catalog description highlighting the product's materials, design, and features in regional {info['language']}>",
  "categories": ["<one or more from the allowed English set>"],
  "tags": ["<exactly 10 descriptive English tags>"],
  "colors": ["<up to 3 primary colors in simple English color names>"]
}}
No extra text or commentary; only return the JSON object."""

    completion = client.chat.completions.create(
        model=vlm_config['model'],
        messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:{content_type};base64,{base64.b64encode(image_bytes).decode()}"}},
            {"type": "text", "text": prompt_text}
        ]}],
        temperature=0.9, top_p=0.9, max_tokens=1024, stream=True
    )

    text = "".join(chunk.choices[0].delta.content for chunk in completion if chunk.choices[0].delta and chunk.choices[0].delta.content)
    logger.info("VLM response received: %d chars", len(text))

    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {"title": "", "description": text, "categories": ["uncategorized"], "tags": [], "colors": []}
    except Exception:
        return {"title": "", "description": text, "categories": ["uncategorized"], "tags": [], "colors": []}

def _call_planner_llm(title: str, description: str, categories: List[str], locale: str = "en-US") -> Dict[str, Any]:
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

Produce ONLY a JSON object with no markdown formatting or code blocks. Example form:
{{"preserve_subject": "white ceramic mug with handle", 
"background_style": "modern kitchen counter, shallow depth of field", 
"camera_angle": "overhead", 
"lighting": "soft natural window light", 
"color_palette": "neutral, warm highlights", 
"negatives": ["do not alter the mug", "no text, no logos, no duplicates", "no changes to color, shape, handle"], 
"cfg_scale": 3.5, "steps": 30, "variants": 1}}

CRITICAL: Return the raw JSON object only - no ```json``` or ``` blocks. Keep the subject unchanged. Do not add extra keys or commentary. Use integers for steps/variants; use a float for cfg_scale between 2.5 and 4.5. Make the background culturally appropriate for {country}."""}
        ],
        temperature=0.6, top_p=1, max_tokens=1024, stream=True
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
    
    return {
        "preserve_subject": title or "product",
        "background_style": "neutral studio cyclorama",
        "camera_angle": "eye-level",
        "lighting": "softbox, high key",
        "color_palette": "neutral",
        "negatives": ["do not alter the subject", "no text, no logos, no duplicates"],
        "cfg_scale": 3.2, "steps": 30, "variants": 1
    }

def _render_flux_prompt(plan: Dict[str, Any]) -> str:
    preserve = plan.get("preserve_subject", "the product")
    background = plan.get("background_style", "neutral studio background")
    camera = plan.get("camera_angle", "eye-level")
    lighting = plan.get("lighting", "softbox")
    negatives = plan.get("negatives", [])
    neg_text = "; ".join(negatives) if isinstance(negatives, list) else str(negatives)
    
    return f"Keep {preserve} unchanged. Replace only the background with {background}. Make it hyperrealistic, ideal for an e-commerce product image. You may include people in the backround with blurring or occlusion. " \
           f"Use {lighting} lighting and {camera} camera angle. " \
           f"Maintain subject color, orientation, and material. Scale the product to natural, proportional size for the environment. " \
           f"{('Avoid: ' + neg_text) if neg_text else ''}".strip()


def _call_flux_edit(image_bytes: bytes, content_type: str, prompt: str, steps: int, cfg_scale: float, seed: Optional[int] = None) -> Dict[str, Any]:
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

    response = requests.post(flux_config['url'], 
        headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json", "Content-Type": "application/json"},
        json={"prompt": prompt, "image": data_url, "aspect_ratio": "match_input_image", 
              "steps": int(steps or 30), "cfg_scale": float(cfg_scale or 3.5), "seed": int(seed if seed is not None else 0)},
        timeout=180)
    
    body = response.json()
    logger.info("FLUX response received: keys=%s", list(body.keys()))
    return body


def _extract_base64_image_from_flux_response(response_body: Dict[str, Any]) -> Optional[str]:
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

def flux_node(state: VLMState) -> VLMState:
    logger.info("FLUX node start")
    image_bytes = state.get("image_bytes")
    prompt = (state.get("flux_prompt") or "").strip()

    if not image_bytes or not prompt:
        error = "image_bytes missing" if not image_bytes else "flux_prompt missing"
        logger.error(f"FLUX node error: {error}")
        return {"error": error, **state}

    try:
        plan = state.get("variation_plan", {})
        flux_response = _call_flux_edit(
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

def vlm_node(state: VLMState) -> VLMState:
    logger.info("VLM node start")
    image_bytes = state.get("image_bytes")
    content_type = state.get("content_type", "image/png")
    locale = state.get("locale", "en-US")
    product_data = state.get("product_data")

    if not image_bytes:
        return {"error": "image_bytes missing or empty", **state}
    if not isinstance(content_type, str) or not content_type.startswith("image/"):
        return {"error": "content_type must be an image/* MIME type", **state}

    vlm_result = _call_vlm(image_bytes, content_type, locale)
    logger.info("VLM analysis: title_len=%d desc_len=%d categories=%s", 
                len(vlm_result.get("title", "")), len(vlm_result.get("description", "")), vlm_result.get("categories", []))
    
    if product_data:
        merged = _call_nemotron_merge(vlm_result, product_data, locale)
        logger.info("Nemotron merge: keys=%s", list(merged.keys()))
        
        categories = (merged.get("categories") if merged.get("categories") and isinstance(merged.get("categories"), list) 
                     else vlm_result.get("categories", ["uncategorized"]))
        
        return {
            **state, 
            "enhanced_product": merged,
            "title": merged.get("title", vlm_result.get("title", "")),
            "description": merged.get("description", vlm_result.get("description", "")),
            "categories": categories,
            "tags": merged.get("tags", vlm_result.get("tags", [])),
            "colors": vlm_result.get("colors", [])
        }
    
    return {
        **state, 
        "title": vlm_result.get("title", ""), 
        "description": vlm_result.get("description", ""), 
        "categories": vlm_result.get("categories", ["uncategorized"]), 
        "tags": vlm_result.get("tags", []), 
        "colors": vlm_result.get("colors", [])
    }

def create_compiled_graph():
    graph = StateGraph(VLMState)
    graph.add_node("vlm", vlm_node)
    graph.add_node("planner", planner_node)
    graph.add_node("flux", flux_node)
    graph.add_node("persist", persist_node)
    graph.set_entry_point("vlm")
    graph.add_edge("vlm", "planner")
    graph.add_edge("planner", "flux")
    graph.add_edge("flux", "persist")
    graph.set_finish_point("persist")
    logger.info("Graph compiled (VLM -> Planner -> FLUX -> Persist): nodes=%s", ["vlm", "planner", "flux", "persist"])
    return graph.compile()
