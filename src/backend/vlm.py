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

def run_vlm_analysis(
    image_bytes: bytes,
    content_type: str,
    locale: str = "en-US",
    product_data: Optional[Dict[str, Any]] = None
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
    
    Returns:
        Dict with title, description, categories, tags, colors, enhanced_product (if augmentation)
    """
    logger.info("Running VLM analysis: locale=%s mode=%s", locale, "augmentation" if product_data else "generation")
    
    if not image_bytes:
        raise ValueError("image_bytes is required")
    if not isinstance(content_type, str) or not content_type.startswith("image/"):
        raise ValueError("content_type must be an image/* MIME type")
    
    # Run VLM analysis
    vlm_result = _call_vlm(image_bytes, content_type, locale)
    logger.info("VLM analysis complete: title_len=%d desc_len=%d categories=%s",
                len(vlm_result.get("title", "")), len(vlm_result.get("description", "")), vlm_result.get("categories", []))
    
    # If product data provided, merge with VLM analysis
    if product_data:
        merged = _call_nemotron_merge(vlm_result, product_data, locale)
        logger.info("Nemotron merge complete: keys=%s", list(merged.keys()))
        
        categories = (merged.get("categories") if merged.get("categories") and isinstance(merged.get("categories"), list)
                     else vlm_result.get("categories", ["uncategorized"]))
        
        return {
            "enhanced_product": merged,
            "title": merged.get("title", vlm_result.get("title", "")),
            "description": merged.get("description", vlm_result.get("description", "")),
            "categories": categories,
            "tags": merged.get("tags", vlm_result.get("tags", [])),
            "colors": vlm_result.get("colors", [])
        }
    
    # Return VLM results directly
    return {
        "title": vlm_result.get("title", ""),
        "description": vlm_result.get("description", ""),
        "categories": vlm_result.get("categories", ["uncategorized"]),
        "tags": vlm_result.get("tags", []),
        "colors": vlm_result.get("colors", [])
    }
