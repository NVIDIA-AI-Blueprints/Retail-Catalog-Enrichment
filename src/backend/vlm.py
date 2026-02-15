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
import re
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
    
    if not (api_key := os.getenv("NGC_API_KEY")):
        raise RuntimeError(NGC_API_KEY_NOT_SET_ERROR)

    info = LOCALE_CONFIG.get(locale, {"language": "English", "region": "United States", "country": "United States", "context": "American English"})
    llm_config = get_config().get_llm_config()
    client = OpenAI(base_url=llm_config['url'], api_key=api_key)

    vlm_json = json.dumps(vlm_output, indent=2, ensure_ascii=False)
    product_json = json.dumps(product_data, indent=2, ensure_ascii=False) if product_data else None

    product_section = f"\nEXISTING PRODUCT DATA:\n{product_json}\n" if product_data else ""

    prompt = f"""/no_think You are a product catalog copywriter. Enhance the content below into compelling e-commerce copy in {info['language']} for {info['region']} ({info['context']}).

VISUAL ANALYSIS (what the camera sees):
{vlm_json}
{product_section}
ALLOWED CATEGORIES: {json.dumps(PRODUCT_CATEGORIES)}

YOUR TASK:
- title: {"The existing title MUST be preserved — keep every word from it and only append visual details (materials, colors, style) to enrich it." if product_data else "Create a compelling product name."} Write in {info['language']}.
- description: Write a rich, persuasive product description highlighting materials, design, and features. {"Expand well beyond the existing description using VLM visual insights." if product_data else "Focus on what makes this product appealing."} Write in {info['language']}.
- categories: Pick from allowed list only. English. Array format.
- tags: {"Keep all existing user tags AND add more from the visual analysis." if product_data else "Generate 10 relevant search tags."} English.
- colors: Use the VLM colors. English.
{f"Keep any other fields from the existing data (price, SKU, etc.) unchanged." if product_data else ""}

Return ONLY valid JSON. No markdown, no comments."""

    logger.info("[Step 1] Sending prompt to Nemotron (length: %d chars)", len(prompt))

    completion = client.chat.completions.create(
        model=llm_config['model'],
        messages=[{"role": "system", "content": "/no_think"}, {"role": "user", "content": prompt}],
        temperature=0.5, top_p=0.9, max_tokens=2048, stream=True,
        extra_body={"reasoning_budget": 16384, "chat_template_kwargs": {"enable_thinking": False}}
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
    
    first_brace = json_text.find('{')
    last_brace = json_text.rfind('}')
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        json_text = json_text[first_brace:last_brace+1]
    
    json_text = re.sub(r'//.*?(?=\n|$)', '', json_text)
    json_text = re.sub(r'/\*.*?\*/', '', json_text, flags=re.DOTALL)

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
    
    if not (api_key := os.getenv("NGC_API_KEY")):
        raise RuntimeError(NGC_API_KEY_NOT_SET_ERROR)

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
        temperature=0.2, top_p=0.9, max_tokens=2048, stream=True,
        extra_body={"reasoning_budget": 16384, "chat_template_kwargs": {"enable_thinking": False}}
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
    
    first_brace = json_text.find('{')
    last_brace = json_text.rfind('}')
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        json_text = json_text[first_brace:last_brace+1]
    
    json_text = re.sub(r'//.*?(?=\n|$)', '', json_text)
    json_text = re.sub(r'/\*.*?\*/', '', json_text, flags=re.DOTALL)

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
    
    api_key = os.getenv("NGC_API_KEY")
    if not api_key:
        raise RuntimeError(NGC_API_KEY_NOT_SET_ERROR)
    
    vlm_config = get_config().get_vlm_config()
    client = OpenAI(base_url=vlm_config['url'], api_key=api_key)

    categories_str = json.dumps(PRODUCT_CATEGORIES)
    
    prompt_text = f"""You are a product catalog copywriter. Analyze the physical product in the image and create compelling e-commerce content.

TASK:
1. Describe the product itself - its materials, design, and features
2. Include any visible brand names or product text shown on the item
3. Write in ENGLISH - be accurate about what you see

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
        temperature=0.3, top_p=0.9, max_tokens=1024, stream=True
    )

    text = "".join(chunk.choices[0].delta.content for chunk in completion if chunk.choices[0].delta and chunk.choices[0].delta.content)
    logger.info("VLM response received: %d chars", len(text))

    json_text = text.strip()
    for marker in ("```json", "```"):
        if marker in json_text:
            try:
                start = json_text.find(marker) + len(marker)
                end = json_text.find("```", start)
                if end > start:
                    json_text = json_text[start:end].strip()
                    break
            except Exception:
                pass

    try:
        parsed = json.loads(json_text)
        return parsed if isinstance(parsed, dict) else {"title": "", "description": json_text, "categories": ["uncategorized"], "tags": [], "colors": []}
    except Exception:
        return {"title": "", "description": json_text, "categories": ["uncategorized"], "tags": [], "colors": []}

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
