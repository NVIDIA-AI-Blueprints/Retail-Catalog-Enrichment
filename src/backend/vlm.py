# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
import base64
import logging
from typing import Optional, Dict, Any

from dotenv import load_dotenv
from openai import OpenAI
from backend.config import get_config
from backend.utils import parse_llm_json

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

# Error messages
NGC_API_KEY_NOT_SET_ERROR = "NGC_API_KEY is not set"

# Allowed product categories for classification
PRODUCT_CATEGORIES = [
    "clothing",
    "footwear",
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

def _call_nemotron_filter_user_data(
    vlm_output: Dict[str, Any],
    product_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Pre-filter: Remove irrelevant terms from user-provided product data before merging.

    Uses a focused, low-temperature LLM call to classify each user-provided term
    as relevant or irrelevant based on the VLM visual analysis (ground truth).
    Returns a cleaned copy of product_data with only relevant terms preserved.
    """
    logger.info("[Pre-filter] Starting relevance filter: vlm_keys=%s, product_keys=%s",
                list(vlm_output.keys()), list(product_data.keys()))

    if not (api_key := os.getenv("NGC_API_KEY")):
        raise RuntimeError(NGC_API_KEY_NOT_SET_ERROR)

    llm_config = get_config().get_llm_config()
    client = OpenAI(base_url=llm_config['url'], api_key=api_key)

    vlm_json = json.dumps(vlm_output, indent=2, ensure_ascii=False)
    product_json = json.dumps(product_data, indent=2, ensure_ascii=False)
    vlm_categories = json.dumps(vlm_output.get("categories", []))

    prompt = f"""You are a product data validator. Decide if the user-provided text is about the SAME type of product shown in the visual analysis, or about a COMPLETELY DIFFERENT product.

VISUAL ANALYSIS (what the camera shows):
{vlm_json}

PRODUCT CATEGORY: {vlm_categories}

USER-PROVIDED PRODUCT DATA:
{product_json}

TASK: For each text field in the user-provided data, answer ONE question: "Is this text about a completely different type of product than what the image shows?"
- If YES (completely different product type, e.g. "laptop" on a shoe image, or "yoga mat" on a blender image) → set that field to an empty string.
- If NO (same product, related, or even partially relevant) → keep the ENTIRE field exactly as the user provided it. Do NOT modify, rephrase, or remove individual words.

This is a binary decision per field — keep it all or clear it all. Never partially edit the user's text.
For non-text fields (price, SKU, numeric values): always keep unchanged.

Return ONLY valid JSON with the same structure as the user-provided data. No markdown, no comments."""

    logger.info("[Pre-filter] Sending filter prompt to Nemotron (length: %d chars)", len(prompt))

    completion = client.chat.completions.create(
        model=llm_config['model'],
        messages=[{"role": "system", "content": ""}, {"role": "user", "content": prompt}],
        temperature=0.1, top_p=0.9, max_tokens=2048, stream=True,
        extra_body={"reasoning_budget": 16384, "chat_template_kwargs": {"enable_thinking": True}}
    )

    text = "".join(chunk.choices[0].delta.content for chunk in completion if chunk.choices[0].delta and chunk.choices[0].delta.content)
    logger.info("[Pre-filter] Nemotron response received: %d chars", len(text))

    parsed = parse_llm_json(text, extract_braces=True, strip_comments=True)
    if parsed is not None:
        logger.info("[Pre-filter] Filter successful: filtered_keys=%s, title_before=%s, title_after=%s",
                    list(parsed.keys()),
                    repr(product_data.get("title", "")),
                    repr(parsed.get("title", "")))
        return parsed
    logger.warning("[Pre-filter] JSON parse failed, using original product data")
    return product_data


def _call_nemotron_enhance_vlm(
    vlm_output: Dict[str, Any],
    product_data: Optional[Dict[str, Any]] = None,
    locale: str = "en-US"
) -> Dict[str, Any]:
    """
    Step 1: Enhance VLM output with compelling copywriting, merge with product data, and localize.

    Receives pre-filtered product_data (irrelevant terms already removed by the
    pre-filter step) and merges it with VLM output into compelling e-commerce copy.
    Includes anti-hallucination rules to prevent fabricating specs not in the input.
    Localizes content to target language/region.
    """
    logger.info("[Step 1] Nemotron enhance + localize: vlm_keys=%s, product_keys=%s, locale=%s", 
                list(vlm_output.keys()), list(product_data.keys()) if product_data else None, locale)
    
    if not (api_key := os.getenv("NGC_API_KEY")):
        raise RuntimeError(NGC_API_KEY_NOT_SET_ERROR)

    info = LOCALE_CONFIG.get(locale, {"language": "English", "region": "United States", "country": "United States", "context": "American English"})
    llm_config = get_config().get_llm_config()
    client = OpenAI(base_url=llm_config['url'], api_key=api_key)

    vlm_json = json.dumps(vlm_output, indent=2, ensure_ascii=False)

    existing_title = product_data.get("title", "") if product_data else ""
    existing_desc = product_data.get("description", "") if product_data else ""

    title_instruction = (
        f'The user provided this title: "{existing_title}". Every word from it MUST appear in the final title. Merge them naturally with visual details from the analysis to create a single compelling product name.'
        if existing_title else "Create a compelling product name."
    )
    desc_instruction = (
        f'The user provided this description: "{existing_desc}". All its words MUST appear in your output — expand around them with visual insights.'
        if existing_desc else "Focus on what makes this product appealing."
    )

    product_section = f"\nEXISTING PRODUCT DATA:\n{json.dumps(product_data, indent=2, ensure_ascii=False)}\n" if product_data else ""

    prompt = f"""/no_think You are a product catalog copywriter. Enhance the content below into compelling e-commerce copy in {info['language']} for {info['region']} ({info['context']}).

VISUAL ANALYSIS (what the camera sees):
{vlm_json}
{product_section}
ALLOWED CATEGORIES: {json.dumps(PRODUCT_CATEGORIES)}

STRICT RULES:
1. NEVER invent or fabricate specific numbers (wattage, capacity, weight, dimensions, HP, voltage, speed counts) on your own. Only use numbers that appear in the VISUAL ANALYSIS or the user-provided title/description above.
2. Numbers from the user-provided title MUST be preserved — they are trusted input, not hallucinations.

YOUR TASK:
- title: {title_instruction} Write in {info['language']}.
- description: Write a rich, persuasive product description highlighting materials, design, and features. Only mention specifications that appear in the visual analysis above. {desc_instruction} Write in {info['language']}.
- categories: Pick from allowed list only. English. Array format.
- tags: {"Keep all existing user tags AND add more from the visual analysis." if product_data else "Generate 10 relevant search tags."} English.
- colors: Use the VLM colors. English.
{f"Keep any other fields from the existing data (price, SKU, etc.) unchanged." if product_data else ""}

Return ONLY valid JSON. No markdown, no comments."""

    logger.info("[Step 1] Sending prompt to Nemotron (length: %d chars)", len(prompt))

    completion = client.chat.completions.create(
        model=llm_config['model'],
        messages=[{"role": "system", "content": "/no_think"}, {"role": "user", "content": prompt}],
        temperature=0.1, top_p=0.9, max_tokens=2048, stream=True,
        extra_body={"reasoning_budget": 16384, "chat_template_kwargs": {"enable_thinking": False}}
    )

    text = "".join(chunk.choices[0].delta.content for chunk in completion if chunk.choices[0].delta and chunk.choices[0].delta.content)
    logger.info("[Step 1] Nemotron response received: %d chars", len(text))

    parsed = parse_llm_json(text, extract_braces=True, strip_comments=True)
    if parsed is not None:
        logger.info("[Step 1] Enhancement successful: enhanced_keys=%s", list(parsed.keys()))
        return parsed
    logger.warning("[Step 1] JSON parse failed, using VLM output")
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
    
    if not (api_key := os.getenv("NGC_API_KEY")):
        raise RuntimeError(NGC_API_KEY_NOT_SET_ERROR)

    info = LOCALE_CONFIG.get(locale, {"language": "English", "region": "United States", "country": "United States", "context": "American English"})
    llm_config = get_config().get_llm_config()
    client = OpenAI(base_url=llm_config['url'], api_key=api_key)

    content_json = json.dumps(enhanced_content, indent=2, ensure_ascii=False)

    prompt = f"""/no_think You are a brand compliance specialist. Apply the following brand-specific instructions to enhance product catalog content.

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

2. **Description Field Formatting** (MANDATORY):
   - CAREFULLY READ the brand instructions for ANY mention of sections, structure, or content types
   - If the brand instructions mention ANY of these, you MUST create clearly labeled sections with headers in the description
   - EVERY section or content type mentioned in the brand instructions MUST appear as a distinct, labeled section in the output - do NOT skip or merge any
   - Each section MUST have a header followed by detailed bullet points or paragraphs
   - CRITICAL: Separate each section with double newlines (\\n\\n) for readability
   - Keep everything in the description field - DO NOT create separate JSON fields for sections
   - The description must be a single string value with proper line breaks between sections
   - When in doubt about whether the brand instructions ask for structure, ALWAYS use structured sections rather than plain prose

3. **Apply Brand Voice** (in {info['language']} for {info['region']}):
   - Apply brand voice/tone to title, description, categories, and tags
   - Use brand-preferred terminology and expressions
   - Maintain factual accuracy while applying brand personality

4. **Categories**:
   - Validate against the allowed categories list above
   - Apply brand taxonomy preferences if specified
   - Keep in English

5. **Tags** (CRITICAL - Preserve User Input):
   - MUST preserve all user-provided tags from the input (do not remove them)
   - ADD brand-preferred terminology and descriptors alongside user tags
   - Keep in English

6. **Preserve All Other Fields**:
   - If enhanced content has fields like price, SKU, colors, specs - preserve them exactly
   - Only modify: title, description, categories, tags

{'═' * 80}
OUTPUT FORMAT:
{'═' * 80}
Return valid JSON with the EXACT SAME structure as the enhanced content input.
Apply brand instructions by modifying the VALUES of existing fields, not by adding new fields.

Return ONLY valid JSON. No markdown, no commentary, no comments (// or /* */)."""

    logger.info("[Step 2] Sending prompt to Nemotron (length: %d chars)", len(prompt))

    completion = client.chat.completions.create(
        model=llm_config['model'],
        messages=[{"role": "system", "content": "/no_think"}, {"role": "user", "content": prompt}],
        temperature=0.1, top_p=0.9, max_tokens=2048, stream=True,
        extra_body={"reasoning_budget": 16384, "chat_template_kwargs": {"enable_thinking": False}}
    )

    text = "".join(chunk.choices[0].delta.content for chunk in completion if chunk.choices[0].delta and chunk.choices[0].delta.content)
    logger.info("[Step 2] Nemotron response received: %d chars", len(text))

    parsed = parse_llm_json(text, extract_braces=True, strip_comments=True)
    if parsed is not None:
        logger.info("[Step 2] Brand alignment successful: keys=%s", list(parsed.keys()))
        return parsed
    logger.warning("[Step 2] JSON parse failed, returning Step 1 content unchanged")
    return enhanced_content


def _call_nemotron_enhance(
    vlm_output: Dict[str, Any], 
    product_data: Optional[Dict[str, Any]] = None,
    locale: str = "en-US", 
    brand_instructions: Optional[str] = None
) -> Dict[str, Any]:
    """
    Orchestrate enhancement pipeline for VLM output.

    Pre-filter (conditional - only if product_data provided):
        - Removes irrelevant terms from user-provided data using category-aware LLM filter

    Step 1: Content enhancement + localization (always runs):
        - Merges pre-filtered product_data with VLM output
        - Applies anti-hallucination rules (no fabricated specs)
        - Localizes to target language/region

    Step 2: Brand alignment (conditional - only if brand_instructions provided):
        - Applies brand voice, tone, taxonomy
    """
    logger.info("Nemotron enhancement pipeline start: vlm_keys=%s, product_keys=%s, locale=%s, brand_instructions=%s", 
                list(vlm_output.keys()), list(product_data.keys()) if product_data else None, locale, bool(brand_instructions))
    
    # Pre-filter: Remove irrelevant terms from user-provided data before merging
    filtered_product_data = product_data
    if product_data:
        filtered_product_data = _call_nemotron_filter_user_data(vlm_output, product_data)
        logger.info("Pre-filter complete: title_before=%s, title_after=%s",
                    repr(product_data.get("title", "")), repr(filtered_product_data.get("title", "")))

    # Step 1: Enhance VLM output and localize to target language (single call for efficiency)
    enhanced = _call_nemotron_enhance_vlm(vlm_output, filtered_product_data, locale)
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
    
    api_key = os.getenv("NGC_API_KEY")
    if not api_key:
        raise RuntimeError(NGC_API_KEY_NOT_SET_ERROR)
    
    vlm_config = get_config().get_vlm_config()
    client = OpenAI(base_url=vlm_config['url'], api_key=api_key)

    categories_str = json.dumps(PRODUCT_CATEGORIES)
    
    prompt_text = f"""You are a product visual analyst. Analyze ONLY what is physically visible in the image. Be strictly factual.

CRITICAL RULES:
- ONLY describe what you can physically SEE in the image.
- Do NOT infer or guess technical specifications (wattage, capacity, weight, dimensions, HP, voltage) unless the exact number is clearly printed and readable on the product label.
- Do NOT fill in details from brand knowledge or training data — if a spec is not visible, do not mention it.
- If you can read text on the product (brand names, labels, buttons), report exactly what is written.

TASK:
1. Describe the product's visible appearance - shape, colors, materials, design elements
2. Transcribe any visible text on the product: brand names, labels, button names, measurements printed on the item
3. Write in ENGLISH - be accurate about what you see, not what you know about the brand

CATEGORIES - Choose ONLY from this allowed set: {categories_str}
- Pick 1-2 categories that GENUINELY describe the product
- It is BETTER to pick only 1 accurate category than to force a second one that doesn't fit
- If only one category applies, return just one: e.g., "categories": ["kitchen"]
- Do NOT stretch or force-fit categories - if the product doesn't belong in a category, don't include it

TAGS: Generate exactly 10 descriptive tags (2-4 words each) for search/filtering

COLORS - What colors would a customer use to describe this product? (1-2 max)
- Include the main material color AND any visible hardware/accent colors (e.g., gold clasps, silver buckles)
- NEVER include the background/backdrop color
- NEVER include hidden parts (shoe soles, inner linings)
- Use simple names: red, blue, black, white, grey, green, yellow, orange, purple, pink, navy, beige, silver, gold, tan, brown, cream, burgundy, olive

Return ONLY valid JSON:
{{
  "title": "<compelling product name>",
  "description": "<detailed catalog description>",
  "categories": ["<category>"],
  "tags": ["<tag1>", "<tag2>", ...],
  "colors": ["<color1>"]
}}"""

    completion = client.chat.completions.create(
        model=vlm_config['model'],
        messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:{content_type};base64,{base64.b64encode(image_bytes).decode()}"}},
            {"type": "text", "text": prompt_text}
        ]}],
        temperature=0.1, top_p=0.9, max_tokens=1024, stream=True
    )

    text = "".join(chunk.choices[0].delta.content for chunk in completion if chunk.choices[0].delta and chunk.choices[0].delta.content)
    logger.info("VLM response received: %d chars", len(text))

    parsed = parse_llm_json(text)
    if parsed is not None:
        return parsed
    return {"title": "", "description": text.strip(), "categories": ["uncategorized"], "tags": [], "colors": []}


def extract_vlm_observation(image_bytes: bytes, content_type: str) -> Dict[str, Any]:
    """Run only the raw VLM observation step."""
    if not image_bytes:
        raise ValueError("image_bytes is required")
    if not isinstance(content_type, str) or not content_type.startswith("image/"):
        raise ValueError("content_type must be an image/* MIME type")

    vlm_result = _call_vlm(image_bytes, content_type)
    logger.info(
        "VLM analysis complete (English): title_len=%d desc_len=%d categories=%s",
        len(vlm_result.get("title", "")),
        len(vlm_result.get("description", "")),
        vlm_result.get("categories", []),
    )
    return vlm_result


def build_enriched_vlm_result(
    vlm_result: Dict[str, Any],
    locale: str = "en-US",
    product_data: Optional[Dict[str, Any]] = None,
    brand_instructions: Optional[str] = None,
) -> Dict[str, Any]:
    """Build enriched catalog fields from a raw VLM observation."""
    enhanced = _call_nemotron_enhance(vlm_result, product_data, locale, brand_instructions)
    logger.info("Nemotron enhance complete: keys=%s", list(enhanced.keys()))

    categories = (
        enhanced.get("categories")
        if enhanced.get("categories") and isinstance(enhanced.get("categories"), list)
        else vlm_result.get("categories", ["uncategorized"])
    )

    result = {
        "title": enhanced.get("title", vlm_result.get("title", "")),
        "description": enhanced.get("description", vlm_result.get("description", "")),
        "categories": categories,
        "tags": enhanced.get("tags", vlm_result.get("tags", [])),
        "colors": enhanced.get("colors", vlm_result.get("colors", [])),
    }

    if product_data:
        result["enhanced_product"] = {**product_data, **enhanced}

    return result

def run_vlm_analysis(
    image_bytes: bytes,
    content_type: str,
    locale: str = "en-US",
    product_data: Optional[Dict[str, Any]] = None,
    brand_instructions: Optional[str] = None,
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
        Dict with title, description, categories, tags, colors, and enhanced_product (if augmentation)
    """
    logger.info("Running VLM analysis: locale=%s mode=%s brand_instructions=%s", locale, "augmentation" if product_data else "generation", bool(brand_instructions))
    vlm_result = extract_vlm_observation(image_bytes, content_type)
    return build_enriched_vlm_result(vlm_result, locale, product_data, brand_instructions)
