import os
import json
import base64
import logging
from typing import TypedDict, Optional, List, Dict, Any

from dotenv import load_dotenv
from openai import OpenAI
from backend.config import get_config

load_dotenv()

logger = logging.getLogger("catalog_enrichment.vlm")

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
    "shoes",
    "skincare"
]

class VLMState(TypedDict, total=False):
    image_bytes: bytes
    content_type: str
    locale: str
    product_data: Optional[Dict[str, Any]]
    brand_instructions: Optional[str]
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

def _call_nemotron_enhance_vlm(
    vlm_output: Dict[str, Any], 
    product_data: Optional[Dict[str, Any]] = None,
    locale: str = "en-US"
) -> Dict[str, Any]:
    """
    Step 1: Enhance VLM output with compelling copywriting and merge with product data.
    
    This function focuses purely on content quality enhancement:
    - Refines raw VLM output with better copywriting
    - Merges with existing product data if provided
    - Applies locale-specific terminology
    - NO brand voice/tone considerations (handled in Step 2)
    """
    logger.info("[Step 1] Nemotron VLM enhance: vlm_keys=%s, product_keys=%s, locale=%s", 
                list(vlm_output.keys()), list(product_data.keys()) if product_data else None, locale)
    
    if not (api_key := os.getenv("NVIDIA_API_KEY")):
        raise RuntimeError("NVIDIA_API_KEY is not set")

    info = LOCALE_CONFIG.get(locale, {"language": "English", "region": "United States", "country": "United States", "context": "American English"})
    llm_config = get_config().get_llm_config()
    client = OpenAI(base_url=llm_config['url'], api_key=api_key)

    vlm_json = json.dumps(vlm_output, indent=2, ensure_ascii=False)
    product_json = json.dumps(product_data, indent=2, ensure_ascii=False) if product_data else None

    # Build the product data section
    product_section = ""
    if product_data:
        product_section = f"""
EXISTING PRODUCT DATA (may contain errors or be incomplete):
{product_json}

"""

    prompt = f"""You are an expert product catalog content specialist for an e-commerce platform. Your role is to create compelling, accurate catalog content.

VISUAL ANALYSIS (from Vision Model - direct observation of the product image):
{vlm_json}
{product_section}
ALLOWED CATEGORIES (must use one or more from this list):
{json.dumps(PRODUCT_CATEGORIES)}

{'═' * 80}
CONTENT ENHANCEMENT STRATEGY:
{'═' * 80}

1. **Title** (in {info['language']} for {info['region']}):
   {'- If existing product data provided:' if product_data else '- Enhance the VLM title with:'}
   {'  * Preserve brand terms, material descriptors, and identifiers from existing title' if product_data else '  * More compelling and precise language'}
   {'  * Incorporate visual details from VLM analysis' if product_data else '  * Focus on key product features'}
   {'  * Create a cohesive result richer than either source' if product_data else '  * Ensure clarity and appeal'}
   - Apply regional terminology: {info['context']}

2. **Description** (in {info['language']} for {info['region']}):
   {'- If existing product data provided:' if product_data else '- Expand the VLM description with:'}
   {'  * Preserve core messaging from existing description' if product_data else '  * Rich, flowing narrative'}
   {'  * Enrich with VLM visual observations' if product_data else '  * Persuasive marketing language'}
   {'  * Create unified, flowing narrative (not concatenation)' if product_data else '  * Focus on materials, design, features'}
   - Use regional language and terminology

3. **Categories** (MUST be an array):
   {'- Validate existing categories against VLM observation' if product_data else '- Use VLM categories as baseline'}
   - MUST use categories from the allowed list above
   - Always return as an array: "categories": ["category1", "category2"]
   - Keep in English

4. **Tags**:
   {'- Combine tags from VLM and existing data, remove duplicates' if product_data else '- Enhance VLM tags with additional relevant terms'}
   - Keep in English
   - Make them specific and useful for search/filtering

5. **Colors**:
   - Use VLM's color analysis (most accurate from visual observation)

{'6. **Trust Priority** (when merging existing product data):' if product_data else ''}
{'   - TRUST visual analysis for: colors, materials, design details, visual attributes' if product_data else ''}
{'   - PRESERVE existing data for: price, SKU, specifications, dimensions' if product_data else ''}
{'   - RESOLVE conflicts by prioritizing visual evidence' if product_data else ''}

{'═' * 80}
OUTPUT FORMAT:
{'═' * 80}
{f'Return the enhanced data using the EXISTING PRODUCT DATA schema/structure. Preserve all original fields and add enriched insights.' if product_data else 'Return enhanced product data with the structure: {"title": "...", "description": "...", "categories": [...], "tags": [...], "colors": [...]}'}

Return ONLY valid JSON with no markdown formatting or commentary."""

    logger.info("[Step 1] Sending prompt to Nemotron (length: %d chars)", len(prompt))

    completion = client.chat.completions.create(
        model=llm_config['model'],
        messages=[{"role": "system", "content": "/no_think"}, {"role": "user", "content": prompt}],
        temperature=0.5, top_p=0.9, max_tokens=2048, stream=True
    )

    text = "".join(chunk.choices[0].delta.content for chunk in completion if chunk.choices[0].delta and chunk.choices[0].delta.content)
    logger.info("[Step 1] Nemotron response received: %d chars", len(text))

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
                logger.warning(f"[Step 1] Failed to extract JSON from {marker}: {e}")

    try:
        parsed = json.loads(json_text)
        if isinstance(parsed, dict):
            logger.info("[Step 1] Enhancement successful: enhanced_keys=%s", list(parsed.keys()))
            return parsed
    except Exception as e:
        logger.warning(f"[Step 1] JSON parse error: {e}, using VLM output")
    
    return vlm_output


def _call_nemotron_apply_branding(
    enhanced_content: Dict[str, Any],
    brand_instructions: str,
    locale: str = "en-US"
) -> Dict[str, Any]:
    """
    Step 2: Apply brand voice, tone, and taxonomy to already-enhanced content.
    
    This function focuses purely on brand alignment:
    - Takes Step 1's enhanced content as input
    - Applies brand-specific voice, tone, and style
    - Applies brand taxonomy and terminology
    - Preserves content quality from Step 1
    """
    logger.info("[Step 2] Nemotron brand application: content_keys=%s, locale=%s", 
                list(enhanced_content.keys()), locale)
    
    if not (api_key := os.getenv("NVIDIA_API_KEY")):
        raise RuntimeError("NVIDIA_API_KEY is not set")

    info = LOCALE_CONFIG.get(locale, {"language": "English", "region": "United States", "country": "United States", "context": "American English"})
    llm_config = get_config().get_llm_config()
    client = OpenAI(base_url=llm_config['url'], api_key=api_key)

    content_json = json.dumps(enhanced_content, indent=2, ensure_ascii=False)

    prompt = f"""You are a brand compliance specialist. Apply the following brand-specific instructions to enhance product catalog content.

BRAND INSTRUCTIONS:
{brand_instructions}

ENHANCED PRODUCT CONTENT (already well-written, needs brand alignment):
{content_json}

ALLOWED CATEGORIES (must use one or more from this list):
{json.dumps(PRODUCT_CATEGORIES)}

{'═' * 80}
CRITICAL RULES:
{'═' * 80}

1. **Maintain Exact JSON Structure**:
   - Return the EXACT SAME JSON keys/fields as the enhanced content above
   - DO NOT add new fields or keys to the JSON
   - DO NOT remove existing fields
   - Only modify the VALUES of existing fields

2. **Description Field Formatting**:
   - If brand instructions specify sections or structure for the description (e.g., "Fragrance Family", "About the Bottle"), incorporate them INTO the description field as formatted text
   - CRITICAL: Separate each section with double newlines (\\n\\n) for readability
   - Each section should be on its own paragraph
   - Format example: "Fragrance Family: ...\\n\\nFragrance Description: ...\\n\\nAbout the Bottle: ...\\n\\nAbout the Fragrance: ..."
   - Keep everything in the description field - DO NOT create separate JSON fields for sections
   - The description must be a single string value with proper line breaks between sections

3. **Apply Brand Voice** (in {info['language']} for {info['region']}):
   - Transform title, description, categories, and tags to match brand voice/tone specified
   - Use brand-preferred terminology and expressions
   - Maintain factual accuracy while applying brand personality

4. **Categories**:
   - Validate against the allowed categories list above
   - Apply brand taxonomy preferences if specified
   - Keep in English

5. **Tags**:
   - Include brand-preferred terminology and descriptors
   - Keep in English

6. **Preserve All Other Fields**:
   - If enhanced content has fields like price, SKU, colors, specs - preserve them exactly
   - Only modify: title, description, categories, tags

{'═' * 80}
OUTPUT FORMAT:
{'═' * 80}
Return valid JSON with the EXACT SAME structure as the enhanced content input.
Apply brand instructions by modifying the VALUES of existing fields, not by adding new fields.

Return ONLY valid JSON with no markdown formatting or commentary."""

    logger.info("[Step 2] Sending prompt to Nemotron (length: %d chars)", len(prompt))

    completion = client.chat.completions.create(
        model=llm_config['model'],
        messages=[{"role": "system", "content": "/no_think"}, {"role": "user", "content": prompt}],
        temperature=0.5, top_p=0.9, max_tokens=2048, stream=True
    )

    text = "".join(chunk.choices[0].delta.content for chunk in completion if chunk.choices[0].delta and chunk.choices[0].delta.content)
    logger.info("[Step 2] Nemotron response received: %d chars", len(text))

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
                logger.warning(f"[Step 2] Failed to extract JSON from {marker}: {e}")

    try:
        parsed = json.loads(json_text)
        if isinstance(parsed, dict):
            logger.info("[Step 2] Brand alignment successful: keys=%s", list(parsed.keys()))
            return parsed
    except Exception as e:
        logger.warning(f"[Step 2] JSON parse error: {e}, returning Step 1 content unchanged")
    
    return enhanced_content


def _call_nemotron_enhance(
    vlm_output: Dict[str, Any], 
    product_data: Optional[Dict[str, Any]] = None,
    locale: str = "en-US", 
    brand_instructions: Optional[str] = None
) -> Dict[str, Any]:
    """
    Orchestrate two-step enhancement pipeline for VLM output.
    
    Step 1: Content enhancement (always runs)
        - Refines VLM output with compelling copywriting
        - Merges with product_data if provided
        - Applies locale-specific terminology
    
    Step 2: Brand alignment (conditional - only if brand_instructions provided)
        - Applies brand voice, tone, and style
        - Applies brand taxonomy and terminology
        - Takes Step 1's output as input
    
    This two-step approach provides clearer context per call and better results.
    """
    logger.info("Nemotron enhancement pipeline start: vlm_keys=%s, product_keys=%s, locale=%s, brand_instructions=%s", 
                list(vlm_output.keys()), list(product_data.keys()) if product_data else None, locale, bool(brand_instructions))
    
    # Step 1: Always enhance VLM output with compelling copywriting
    enhanced = _call_nemotron_enhance_vlm(vlm_output, product_data, locale)
    logger.info("Step 1 complete: enhanced_keys=%s", list(enhanced.keys()))
    
    # Step 2: Apply brand instructions if provided
    if brand_instructions:
        enhanced = _call_nemotron_apply_branding(enhanced, brand_instructions, locale)
        logger.info("Step 2 complete: brand-aligned content ready")
    else:
        logger.info("Step 2 skipped: no brand_instructions provided")
    
    logger.info("Nemotron enhancement pipeline complete: final_keys=%s", list(enhanced.keys()))
    return enhanced

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

READ AND INCORPORATE VISIBLE TEXT:
- Carefully read any text visible on the product packaging, labels, or the product itself
- Look for and include: net weight (e.g., "Net Wt. 50g"), volume (e.g., "250ml", "8 fl oz"), dimensions, quantities, product names, brand names, variant descriptions (e.g., "Moisturizing", "Extra Strength")
- Incorporate these specific details naturally into the title and description
- If measurements use both metric and imperial units, include both
- Use the exact text visible for product names or variant types to ensure accuracy
- Example: If you see "Net Wt. 1.7 oz / 50g", include "1.7 oz (50g)" in the description

Create an engaging product title and description in {info['language']} as spoken in {info['region']}. {info['context']}.

Classify the product into one or more categories from this fixed allowed set only:
{categories_str}
If none apply, use ["uncategorized"].

Generate exactly 10 useful product tags that describe the item's characteristics, materials, style, features, or type. These should be short descriptive phrases (2-4 words each) that would help customers find this product. Use English for the tags.

Extract up to 3 primary colors visible in the product. Choose the most prominent and distinctive colors that would help customers identify or search for the product. Use simple, standard color names in English (e.g., "red", "blue", "black", "white", "brown", "grey", "green", "yellow", "orange", "purple", "pink", "navy", "beige", "cream", "silver", "gold"). If fewer than 3 distinct colors are clearly visible, provide only the colors you can confidently identify.

IMPORTANT GUIDELINES:
- Write compelling catalog copy that sells the physical product itself
- Highlight the product's materials, design, construction, and notable features
- Incorporate any visible specifications (weight, volume, dimensions) from text on the product
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

def vlm_node(state: VLMState) -> VLMState:
    logger.info("VLM node start")
    image_bytes = state.get("image_bytes")
    content_type = state.get("content_type", "image/png")
    locale = state.get("locale", "en-US")
    product_data = state.get("product_data")
    brand_instructions = state.get("brand_instructions")

    if not image_bytes:
        return {"error": "image_bytes missing or empty", **state}
    if not isinstance(content_type, str) or not content_type.startswith("image/"):
        return {"error": "content_type must be an image/* MIME type", **state}

    vlm_result = _call_vlm(image_bytes, content_type, locale)
    logger.info("VLM analysis: title_len=%d desc_len=%d categories=%s", 
                len(vlm_result.get("title", "")), len(vlm_result.get("description", "")), vlm_result.get("categories", []))
    
    # Always enhance VLM output with Nemotron (handles all scenarios)
    enhanced = _call_nemotron_enhance(vlm_result, product_data, locale, brand_instructions)
    logger.info("Nemotron enhance: keys=%s", list(enhanced.keys()))
    
    categories = (enhanced.get("categories") if enhanced.get("categories") and isinstance(enhanced.get("categories"), list) 
                 else vlm_result.get("categories", ["uncategorized"]))
    
    result = {
        **state, 
        "title": enhanced.get("title", vlm_result.get("title", "")),
        "description": enhanced.get("description", vlm_result.get("description", "")),
        "categories": categories,
        "tags": enhanced.get("tags", vlm_result.get("tags", [])),
        "colors": enhanced.get("colors", vlm_result.get("colors", []))
    }
    
    # If product data was provided, merge enhanced fields back into original product_data
    # to preserve untouched fields (price, SKU, specs, etc.)
    if product_data:
        result["enhanced_product"] = {**product_data, **enhanced}
    
    return result

def run_vlm_analysis(
    image_bytes: bytes,
    content_type: str,
    locale: str = "en-US",
    product_data: Optional[Dict[str, Any]] = None,
    brand_instructions: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run VLM analysis on an image to extract product fields.
    
    This is a standalone function that runs only the VLM analysis
    (without image generation).
    
    Args:
        image_bytes: Product image bytes
        content_type: Image MIME type
        locale: Target locale for analysis
        product_data: Optional existing product data to augment
        brand_instructions: Optional brand-specific tone/style instructions
    
    Returns:
        Dict with title, description, categories, tags, colors, enhanced_product (if augmentation)
    """
    logger.info("Running VLM analysis: locale=%s mode=%s brand_instructions=%s", locale, "augmentation" if product_data else "generation", bool(brand_instructions))
    
    if not image_bytes:
        raise ValueError("image_bytes is required")
    if not isinstance(content_type, str) or not content_type.startswith("image/"):
        raise ValueError("content_type must be an image/* MIME type")
    
    # Run VLM analysis
    vlm_result = _call_vlm(image_bytes, content_type, locale)
    logger.info("VLM analysis complete: title_len=%d desc_len=%d categories=%s",
                len(vlm_result.get("title", "")), len(vlm_result.get("description", "")), vlm_result.get("categories", []))
    
    # Always enhance VLM output with Nemotron (handles all scenarios)
    enhanced = _call_nemotron_enhance(vlm_result, product_data, locale, brand_instructions)
    logger.info("Nemotron enhance complete: keys=%s", list(enhanced.keys()))
    
    categories = (enhanced.get("categories") if enhanced.get("categories") and isinstance(enhanced.get("categories"), list)
                 else vlm_result.get("categories", ["uncategorized"]))
    
    result = {
        "title": enhanced.get("title", vlm_result.get("title", "")),
        "description": enhanced.get("description", vlm_result.get("description", "")),
        "categories": categories,
        "tags": enhanced.get("tags", vlm_result.get("tags", [])),
        "colors": enhanced.get("colors", vlm_result.get("colors", []))
    }
    
    # If product data was provided, merge enhanced fields back into original product_data
    # to preserve untouched fields (price, SKU, specs, etc.)
    if product_data:
        result["enhanced_product"] = {**product_data, **enhanced}
    
    return result
