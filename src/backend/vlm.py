import os
import json
import base64
import logging
from typing import Optional, List, Dict, Any

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
    "fragrance",
    "skincare",
    "bags"
]

def _call_nemotron_enhance_vlm(
    vlm_output: Dict[str, Any], 
    product_data: Optional[Dict[str, Any]] = None,
    locale: str = "en-US"
) -> Dict[str, Any]:
    """
    Step 1: Enhance VLM output with compelling copywriting, merge with product data, and localize.
    
    This function handles:
    - Refines raw VLM output (which is always in English) with better copywriting
    - Merges with existing product data if provided
    - Localizes content to target language/region (done here to avoid extra LLM call)
    - NO brand voice/tone considerations (handled in Step 2)
    """
    logger.info("[Step 1] Nemotron enhance + localize: vlm_keys=%s, product_keys=%s, locale=%s", 
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

    prompt = f"""/no_think You are an expert product catalog content specialist for an e-commerce platform. Your role is to create compelling, accurate catalog content.

VISUAL ANALYSIS (from Vision Model - always in English, direct observation of the product image):
{vlm_json}
{product_section}
ALLOWED CATEGORIES (must use one or more from this list):
{json.dumps(PRODUCT_CATEGORIES)}

{'═' * 80}
CONTENT ENHANCEMENT & LOCALIZATION STRATEGY:
{'═' * 80}

IMPORTANT: The VLM analysis above is always in English. Your task is to enhance it and then localize to {info['language']} for {info['region']}.

1. **Title** (in {info['language']} for {info['region']}):
   {'- If existing product data provided:' if product_data else '- Enhance the VLM title with:'}
   {'  * Preserve brand terms, material descriptors, and identifiers from existing title' if product_data else '  * More compelling and precise language'}
   {'  * Incorporate visual details from VLM analysis' if product_data else '  * Focus on key product features'}
   {'  * Create a cohesive result richer than either source' if product_data else '  * Ensure clarity and appeal'}
   - Translate naturally to {info['language']} using proper regional terminology: {info['context']}
   - Maintain accuracy - don't hallucinate product types

2. **Description** (in {info['language']} for {info['region']}):
   {'- If existing product data provided:' if product_data else '- Expand the VLM description with:'}
   {'  * Preserve core messaging from existing description' if product_data else '  * Rich, flowing narrative'}
   {'  * Enrich with VLM visual observations' if product_data else '  * Persuasive marketing language'}
   {'  * Create unified, flowing narrative (not concatenation)' if product_data else '  * Focus on materials, design, features'}
   - Translate naturally to {info['language']} with regional terminology
   - Maintain factual accuracy from the VLM observation

3. **Categories** (MUST be an array):
   {'- Validate existing categories against VLM observation' if product_data else '- Use VLM categories as baseline'}
   - MUST use categories from the allowed list above
   - Always return as an array: "categories": ["category1", "category2"]
   - Keep in English (not translated)

4. **Tags**:
   {'- Combine tags from VLM and existing data, remove duplicates' if product_data else '- Enhance VLM tags with additional relevant terms'}
   - Keep in English (not translated)
   - Make them specific and useful for search/filtering

5. **Colors**:
   - Use VLM's color analysis (most accurate from visual observation)
   - Keep in English (not translated)

{'6. **Trust Priority** (when merging existing product data):' if product_data else ''}
{'   - TRUST visual analysis for: colors, materials, design details, visual attributes' if product_data else ''}
{'   - PRESERVE existing data for: price, SKU, specifications, dimensions' if product_data else ''}
{'   - RESOLVE conflicts by prioritizing visual evidence' if product_data else ''}

{'═' * 80}
OUTPUT FORMAT:
{'═' * 80}
{f'Return the enhanced data using the EXISTING PRODUCT DATA schema/structure. Preserve all original fields and add enriched insights. Translate title and description to {info["language"]}.' if product_data else f'Return enhanced product data with the structure: {{"title": "...", "description": "...", "categories": [...], "tags": [...], "colors": [...]}}. Write title and description in {info["language"]} for {info["region"]}.'}

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
   - If brand instructions specify sections or structure for the description, incorporate them INTO the description field as formatted text
   - CRITICAL: Separate each section with double newlines (\\n\\n) for readability
   - Each section should be on its own paragraph
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
    
    Step 1: Content enhancement + localization (always runs)
        - Refines VLM output (which is always in English) with compelling copywriting
        - Merges with product_data if provided
        - Localizes to target language/region (single LLM call for efficiency)
    
    Step 2: Brand alignment (conditional - only if brand_instructions provided)
        - Applies brand voice, tone, and style
        - Applies brand taxonomy and terminology
        - Takes Step 1's output as input
    
    This approach ensures VLM works only in English (preventing hallucinations),
    while LLM handles accurate localization and enhancement.
    """
    logger.info("Nemotron enhancement pipeline start: vlm_keys=%s, product_keys=%s, locale=%s, brand_instructions=%s", 
                list(vlm_output.keys()), list(product_data.keys()) if product_data else None, locale, bool(brand_instructions))
    
    # Step 1: Enhance VLM output and localize to target language (single call for efficiency)
    enhanced = _call_nemotron_enhance_vlm(vlm_output, product_data, locale)
    logger.info("Step 1 complete (enhanced + localized to %s): enhanced_keys=%s", locale, list(enhanced.keys()))
    
    # Step 2: Apply brand instructions if provided
    if brand_instructions:
        enhanced = _call_nemotron_apply_branding(enhanced, brand_instructions, locale)
        logger.info("Step 2 complete: brand-aligned content ready")
    else:
        logger.info("Step 2 skipped: no brand_instructions provided")
    
    logger.info("Nemotron enhancement pipeline complete: final_keys=%s", list(enhanced.keys()))
    return enhanced

def _call_vlm(image_bytes: bytes, content_type: str) -> Dict[str, Any]:
    """Call VLM to analyze product image.
    
    NOTE: Always analyzes in ENGLISH regardless of target locale.
    This prevents hallucinations that occur when VLMs work in non-English languages.
    Localization is handled separately by the LLM in a subsequent step.
    """
    logger.info("Calling VLM: bytes=%d, content_type=%s (English-only analysis)", len(image_bytes or b""), content_type)
    
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError("NVIDIA_API_KEY is not set")
    
    vlm_config = get_config().get_vlm_config()
    client = OpenAI(base_url=vlm_config['url'], api_key=api_key)

    categories_str = json.dumps(PRODUCT_CATEGORIES)
    
    prompt_text = f"""You are a product catalog copywriter for an e-commerce platform. Create compelling catalog content for the physical product shown in the image. Be verbose and detailed.

Focus on the actual product visible in the image - describe the item itself, its materials, design, and features. Do not focus on contents, intended use scenarios, or lifestyle experiences.

READ AND INCORPORATE VISIBLE TEXT (IF PRESENT):
- Look for brand names, product names, or variant descriptions visible on the product or packaging (e.g., "Moisturizing", "Extra Strength", "Matte Finish")
- Use the exact text visible to ensure accuracy

CRITICAL: Write all content in ENGLISH. Be accurate and precise about what you see.

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
- Be specific and accurate - for example, distinguish between a handbag, beach bag, tote, clutch, etc.
- Keep all content in ENGLISH (title, description, categories, tags, colors)
- Make tags specific and useful for search/filtering

Return ONLY valid JSON with the following structure:
{{
  "title": "<compelling product name in ENGLISH describing what you see>",
  "description": "<persuasive catalog description in ENGLISH with accurate product details>",
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
    
    # Run VLM analysis (always in English)
    vlm_result = _call_vlm(image_bytes, content_type)
    logger.info("VLM analysis complete (English): title_len=%d desc_len=%d categories=%s",
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
