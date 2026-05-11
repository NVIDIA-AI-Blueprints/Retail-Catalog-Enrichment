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
    "toys",
    "electronics",
    "furniture",
    "office",
    "skincare",
    "bags",
    "outdoor"
]
FALLBACK_CATEGORY = "uncategorized"
CATEGORY_OUTPUT_VALUES = PRODUCT_CATEGORIES + [FALLBACK_CATEGORY]
CATEGORY_OUTPUT_SET = frozenset(CATEGORY_OUTPUT_VALUES)

ALLOWED_COLORS = [
    "black",
    "white",
    "gray",
    "silver",
    "gold",
    "brown",
    "beige",
    "red",
    "orange",
    "yellow",
    "green",
    "blue",
    "purple",
    "pink",
]
COLOR_ALIASES = {"grey": "gray"}
ALLOWED_COLOR_SET = frozenset(ALLOWED_COLORS)

LOCALIZED_TERMINOLOGY_RULE = (
    "Use established retail terminology for the target locale in localized customer-facing fields. "
    "The visual analysis may be in English; translate generic product-type nouns from the visual analysis into natural, widely used terms in the target language. "
    "English generic product-type nouns are not allowed in localized title or description output. "
    "Do not keep English generic product-type nouns just because they appear in the visual analysis or as readable label text. "
    "Do not invent new compound words, calques, or phonetic translations; never coin or merge words to translate a product type. "
    "If unsure, use a common generic product term in the target language instead of inventing one. "
    "Keep English only for brand names, model names, readable printed text, or terms explicitly provided as official product names; readable English label text does not override the localized generic product type. "
    "Before returning JSON, self-check title and description; if an English generic product-type noun remains, translate it into the target language. "
    "Use the chosen product-type term consistently across localized customer-facing fields."
)


def _localized_terminology_rule(info: Dict[str, str]) -> str:
    """Return terminology guard only when the target output is not English."""
    if info.get("language") == "English":
        return ""
    return LOCALIZED_TERMINOLOGY_RULE


def _localized_terminology_block(info: Dict[str, str]) -> str:
    """Return a prominent localization check for non-English catalog generation."""
    rule = _localized_terminology_rule(info)
    if not rule:
        return ""
    return f"""
LOCALIZATION CHECK:
- {rule}
- Title and description are invalid if they keep English generic product-type nouns for the product type.
- Before returning JSON, rewrite any remaining English generic product-type noun into {info['language']} while keeping brand/model names unchanged."""


def _normalize_categories(categories: Any) -> list[str]:
    """Keep only supported category labels and preserve first-seen order."""
    if not isinstance(categories, list):
        return []

    normalized = []
    for value in categories:
        if not isinstance(value, str):
            continue
        category = value.strip().lower()
        if category in CATEGORY_OUTPUT_SET and category not in normalized:
            normalized.append(category)

    if len(normalized) > 1 and FALLBACK_CATEGORY in normalized:
        return [category for category in normalized if category != FALLBACK_CATEGORY]
    return normalized


def _normalize_colors(colors: Any) -> list[str]:
    """Keep only generic color names, not materials/finishes."""
    if not isinstance(colors, list):
        return []

    normalized = []
    for value in colors:
        if not isinstance(value, str):
            continue
        for word in re.findall(r"[a-z]+", value.lower()):
            color = COLOR_ALIASES.get(word, word)
            if color in ALLOWED_COLOR_SET and color not in normalized:
                normalized.append(color)
    return normalized


def _call_nemotron_filter_user_data(
    vlm_output: Dict[str, Any],
    product_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Pre-filter: Remove irrelevant or contradictory user terms before merging.

    Uses a focused, low-temperature LLM call to clean user-provided text against
    the VLM visual analysis. Readable label text is treated as ground truth for
    product names, variants, active ingredients, formulations, counts, and specs.
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

    prompt = f"""You are a product data validator. Clean user-provided product data before it is merged with visual analysis.

The VISUAL ANALYSIS is ground truth for visible facts and readable label text. User-provided data may contain stale, copied, or partially wrong terms.

VISUAL ANALYSIS (what the camera shows):
{vlm_json}

PRODUCT CATEGORY: {vlm_categories}

USER-PROVIDED PRODUCT DATA:
{product_json}

TASK:
- Return the same JSON structure after removing user-provided text that conflicts with the visual analysis.
- Preserve non-conflicting user evidence, including brand names, model names, SKU, price, materials, and internal specs that are not visibly contradicted.
- If a text field is about a completely different product type, set that field to an empty string.
- If a text field is partially correct, edit that field minimally: keep correct terms and remove only the conflicting terms.
- Readable label text is authoritative for product names, active ingredients, flavors, scents, colors, variants, formulations, package counts, dosages, ratings, and model/spec values.
- If the user-provided product name, active ingredient, flavor, scent, variant, formulation, count, dosage, rating, or model/spec value differs from readable label text or the visually identified product type, remove the user-provided conflicting term.
- Do not combine two conflicting product identities into one title or description. Use the visual/readable-label identity and any non-conflicting user terms.
- Do not replace a conflicting user term with a new term unless that replacement is directly present in the visual analysis; otherwise remove the conflicting term and let the later enrichment step fill from visual evidence.

For non-text fields (price, SKU, numeric values): always keep unchanged.

Return ONLY valid JSON with the same structure as the user-provided data. No markdown, no comments."""

    logger.info("[Pre-filter] Sending filter prompt to Nemotron (length: %d chars)", len(prompt))

    completion = client.chat.completions.create(
        model=llm_config['model'],
        messages=[{"role": "system", "content": ""}, {"role": "user", "content": prompt}],
        temperature=0.1, top_p=0.9, max_tokens=2048, stream=True,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}}
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
    localized_terminology_rule = _localized_terminology_rule(info)
    localized_terminology_line = f"9. {localized_terminology_rule}" if localized_terminology_rule else ""
    llm_config = get_config().get_llm_config()
    client = OpenAI(base_url=llm_config['url'], api_key=api_key)

    vlm_json = json.dumps(vlm_output, indent=2, ensure_ascii=False)

    existing_title = product_data.get("title", "") if product_data else ""
    existing_desc = product_data.get("description", "") if product_data else ""

    if existing_title and localized_terminology_rule:
        title_instruction = (
            f'The user provided this title after contradiction filtering: "{existing_title}". Treat the remaining user title terms as validated anchors, not as complete truth. Preserve the user\'s product intent plus any brand/model names and factual specs when they do not conflict with the VISUAL ANALYSIS. Localize common product-type words using established retail terminology when needed. Do not replace user factual words with unrelated synonyms. Add only title-worthy facts from the VISUAL ANALYSIS: official product name, product type, variant, flavor, scent, formulation, count, dosage, rating, size, material, compatibility, or model/spec values when readable/provided. Do not add packaging/container appearance such as cap color, bottle color, box color, label color, banner color, background color, shape, or label placement to the title unless it is an official product variant or necessary to distinguish the sold product. If readable label text identifies a product name, active ingredient, flavor, scent, variant, formulation, count, dosage, rating, or model/spec value, it overrides any remaining conflicting user title term. Do not combine conflicting product identities in the final title. If the visual analysis has useful title-worthy details, the final title must be more specific than, and not identical to, the user-provided title. Do not invent materials/specs; keep user material/spec terms when present.'
        )
    elif existing_title:
        title_instruction = (
            f'The user provided this title after contradiction filtering: "{existing_title}". Treat the remaining user title terms as validated anchors, not as complete truth. Preserve product names, brand/model names, and factual specs when they do not conflict with the VISUAL ANALYSIS. Do not replace user title words with unrelated synonyms. Add only title-worthy facts from the VISUAL ANALYSIS: official product name, product type, variant, flavor, scent, formulation, count, dosage, rating, size, material, compatibility, or model/spec values when readable/provided. Do not add packaging/container appearance such as cap color, bottle color, box color, label color, banner color, background color, shape, or label placement to the title unless it is an official product variant or necessary to distinguish the sold product. If readable label text identifies a product name, active ingredient, flavor, scent, variant, formulation, count, dosage, rating, or model/spec value, it overrides any remaining conflicting user title term. Do not combine conflicting product identities in the final title. If the visual analysis has useful title-worthy details, the final title must be more specific than, and not identical to, the user-provided title. Do not invent materials/specs; keep user material/spec terms when present.'
        )
    else:
        title_instruction = "Create a compelling product name."
    desc_instruction = (
        f'The user provided this description: "{existing_desc}". Use it as the BASE and expand it with visual details from the analysis. Keep all user terms unless printed label text on the product clearly contradicts them.'
        if existing_desc else "Focus on what makes this product appealing."
    )

    product_section = f"\nEXISTING PRODUCT DATA:\n{json.dumps(product_data, indent=2, ensure_ascii=False)}\n" if product_data else ""

    prompt = f"""/no_think You are a product catalog copywriter. Enhance the content below into compelling e-commerce copy in {info['language']} for {info['region']} ({info['context']}).

VISUAL ANALYSIS (what the camera sees):
{vlm_json}
{product_section}
ALLOWED CATEGORIES: {json.dumps(CATEGORY_OUTPUT_VALUES)}
ALLOWED COLORS: {json.dumps(ALLOWED_COLORS)}

STRICT RULES:
1. NEVER invent or fabricate details on your own. Only use facts from the VISUAL ANALYSIS or the EXISTING PRODUCT DATA above.
2. Printed text readable on the product (brand names, product names, dosages, model numbers) is ground truth. Drop user words that contradict printed label text.
3. Material descriptions from the visual analysis are visual guesses — the camera cannot verify composition. Always use the user's material term when provided.
4. The VISUAL ANALYSIS is authoritative for appearance (colors, shape, design) and printed text. The EXISTING PRODUCT DATA is authoritative for material composition and internal specs.
5. {"In augmentation mode, filtered user-provided title words are validated anchors: keep them when natural for the target locale and not visibly contradicted, localize common product-type terms when needed, and add only title-worthy product identity/spec facts around them." if localized_terminology_rule else "In augmentation mode, filtered user-provided title words are validated anchors: keep them when not visibly contradicted, then add only title-worthy product identity/spec facts around them."}
6. Do not state measurable specs such as capacity, dimensions, volume, weight, power rating, counts, compatibility, or model/spec values unless they are readable in the image or explicitly provided by the user.
7. Do not use size/weight claims such as compact, large, spacious, lightweight, or heavy unless scale is visible or the user provided that detail.
8. Colors must be selected from ALLOWED COLORS only. Do not output materials, finishes, textures, or product types as colors; choose the closest visible generic color instead.
9. Do not include packaging/container appearance in titles: cap color, bottle color, box color, label color, banner color, background color, shape, label placement, or similar visual packaging details belong in description/tags, not title, unless they are official product variants or necessary retail differentiators.
{localized_terminology_line}

YOUR TASK:
- title: {title_instruction} Write in {info['language']}.
- description: Write a rich, persuasive product description. Merge visual details with user-provided information. {desc_instruction} Write in {info['language']}.
- categories: Pick from allowed list only. English. Array format.
- tags: {"Keep all existing user tags AND add more from the visual analysis." if product_data else "Generate 10 relevant search tags."} English.
- colors: Use visible product colors from ALLOWED COLORS only. English.
{f"Keep any other fields from the existing data (price, SKU, etc.) unchanged." if product_data else ""}

Return ONLY valid JSON. No markdown, no comments."""

    logger.info("[Step 1] Sending prompt to Nemotron (length: %d chars)", len(prompt))

    completion = client.chat.completions.create(
        model=llm_config['model'],
        messages=[{"role": "system", "content": "/no_think"}, {"role": "user", "content": prompt}],
        temperature=0.1, top_p=0.9, max_tokens=2048, stream=True,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}}
    )

    text = "".join(chunk.choices[0].delta.content for chunk in completion if chunk.choices[0].delta and chunk.choices[0].delta.content)
    logger.info("[Step 1] Nemotron response received: %d chars", len(text))

    parsed = parse_llm_json(text, extract_braces=True, strip_comments=True)
    if parsed is not None:
        logger.info("[Step 1] Enhancement successful: enhanced_keys=%s", list(parsed.keys()))
        return parsed
    logger.warning("[Step 1] JSON parse failed, using VLM output")
    return vlm_output


def _call_nemotron_resolve_merge_conflicts(
    vlm_output: Dict[str, Any],
    product_data: Dict[str, Any],
    merged_content: Dict[str, Any],
    locale: str = "en-US",
) -> Dict[str, Any]:
    """Remove contradictions that survive the initial user-data merge."""
    logger.info("[Merge QA] Resolving merge conflicts: merged_keys=%s, locale=%s", list(merged_content.keys()), locale)

    if not (api_key := os.getenv("NGC_API_KEY")):
        raise RuntimeError(NGC_API_KEY_NOT_SET_ERROR)

    info = LOCALE_CONFIG.get(locale, {"language": "English", "region": "United States", "country": "United States", "context": "American English"})
    llm_config = get_config().get_llm_config()
    client = OpenAI(base_url=llm_config['url'], api_key=api_key)

    prompt = f"""/no_think You are a product catalog merge QA validator. Review the merged catalog content and remove contradictions between user-provided data and visual/readable-label evidence.

VISUAL ANALYSIS (ground truth for visible facts and readable label text):
{json.dumps(vlm_output, indent=2, ensure_ascii=False)}

FILTERED USER DATA (non-conflicting user evidence that may still be incomplete):
{json.dumps(product_data, indent=2, ensure_ascii=False)}

MERGED CATALOG CONTENT TO VALIDATE:
{json.dumps(merged_content, indent=2, ensure_ascii=False)}

TARGET LANGUAGE / REGION: {info['language']} ({info['region']}, {info['context']})

RULES:
- Return the exact same JSON keys as MERGED CATALOG CONTENT. Do not add or remove fields.
- Preserve non-conflicting user evidence such as brand names, model names, SKU, price, materials, and internal specs.
- Readable label text is authoritative for product names, active ingredients, flavors, scents, colors, variants, formulations, package counts, dosages, ratings, and model/spec values.
- If the merged title, description, categories, tags, or enhanced_product contains a user-derived product identity term that conflicts with readable label text or the visually identified product type, remove it or replace it with the exact supported visual/readable-label term.
- Do not combine two conflicting product identities in the title, description, tags, or enhanced_product.
- Do not remove a term merely because it is absent from the image; remove it only when it conflicts with the visual/readable-label identity.
- Title should contain customer-facing product identity only: brand, official product name, product type, variant, flavor, scent, formulation, count, dosage, rating, size, material, compatibility, and model/spec values when supported. Remove packaging/container appearance from title, such as cap color, bottle color, box color, label color, banner color, background color, shape, or label placement, unless it is an official variant or necessary retail differentiator.
- Keep the output in {info['language']} for customer-facing title and description.

Return ONLY valid JSON. No markdown, no comments."""

    completion = client.chat.completions.create(
        model=llm_config['model'],
        messages=[{"role": "system", "content": "/no_think"}, {"role": "user", "content": prompt}],
        temperature=0.0, top_p=1, max_tokens=2048, stream=True,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}}
    )

    text = "".join(
        chunk.choices[0].delta.content
        for chunk in completion
        if chunk.choices[0].delta and chunk.choices[0].delta.content
    )
    logger.info("[Merge QA] Nemotron response received: %d chars", len(text))

    parsed = parse_llm_json(text, extract_braces=True, strip_comments=True)
    if isinstance(parsed, dict):
        logger.info("[Merge QA] Conflict validation complete: keys=%s", list(parsed.keys()))
        return parsed

    logger.warning("[Merge QA] JSON parse failed, keeping merged content unchanged")
    return merged_content


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
    localized_terminology_rule = _localized_terminology_rule(info)
    localized_terminology_bullet = f"- {localized_terminology_rule}" if localized_terminology_rule else ""
    llm_config = get_config().get_llm_config()
    client = OpenAI(base_url=llm_config['url'], api_key=api_key)

    content_json = json.dumps(enhanced_content, indent=2, ensure_ascii=False)

    prompt = f"""/no_think You are a brand compliance specialist. Apply the following brand-specific instructions to enhance product catalog content.

OUTPUT LANGUAGE LOCK:
- Title and description must remain in {info['language']} for {info['region']} ({info['context']}).
- Brand instructions may be written in any language. Treat them only as style guidance; do not infer the output language from them.
- Do not output title or description in any language other than {info['language']}.
{localized_terminology_bullet}

BRAND INSTRUCTIONS:
{brand_instructions}

ENHANCED PRODUCT CONTENT (already well-written, needs brand alignment):
{content_json}

ALLOWED CATEGORIES (must use one or more from this list):
{json.dumps(CATEGORY_OUTPUT_VALUES)}

{'═' * 80}
CRITICAL RULES:
{'═' * 80}

1. **Maintain Exact JSON Structure**:
   - Return the EXACT SAME JSON keys/fields as the enhanced content above
   - DO NOT add new fields or keys to the JSON
   - DO NOT remove existing fields
   - Only modify the VALUES of existing fields

2. **Description Field Formatting**:
   - Follow the brand instructions for format and structure — if they ask for paragraphs, write paragraphs; if they ask for sections or bullet points, use sections and bullet points
   - If brand instructions ask for a richer, longer, more detailed, premium, luxurious, elevated, or more persuasive description, expand the description using only product facts and visible/design details already present in the enhanced content. Add 1-3 additional sentences when enough source detail exists.
   - Keep everything in the description field as a single string value
   - Separate sections or paragraphs with double newlines (\\n\\n) for readability

3. **Apply Brand Voice** (in {info['language']} for {info['region']}):
   - Apply brand voice/tone to title and description while preserving the target language above
   - Use brand-preferred terminology and expressions
   - Do NOT add ingredients, specifications, or features not present in the enhanced content above. Rephrase, style, and when requested, safely expand only what is already there

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
   - Do NOT add new measurable specs such as capacity, dimensions, volume, weight, power rating, counts, compatibility, or model/spec values
   - Do NOT add size/weight claims such as compact, large, spacious, lightweight, or heavy unless they already appear in the enhanced content

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
        extra_body={"chat_template_kwargs": {"enable_thinking": False}}
    )

    text = "".join(chunk.choices[0].delta.content for chunk in completion if chunk.choices[0].delta and chunk.choices[0].delta.content)
    logger.info("[Step 2] Nemotron response received: %d chars", len(text))

    parsed = parse_llm_json(text, extract_braces=True, strip_comments=True)
    if parsed is not None:
        logger.info("[Step 2] Brand alignment successful: keys=%s", list(parsed.keys()))
        return parsed
    logger.warning("[Step 2] JSON parse failed, returning Step 1 content unchanged")
    return enhanced_content


def _format_manual_knowledge(knowledge: Dict[str, str]) -> str:
    """Format extracted manual knowledge into a prompt section."""
    lines = ["PRODUCT MANUAL KNOWLEDGE:",
             "The following information was extracted from the official product manual.\n"]
    for topic, content in knowledge.items():
        label = topic.replace("_", " ").title()
        if content and content.strip():
            lines.append(f"[{label}]")
            lines.append(content.strip())
            lines.append("")
    return "\n".join(lines)


def _call_nemotron_generate_faqs(
    enriched_result: Dict[str, Any],
    locale: str = "en-US",
    manual_knowledge: Optional[Dict[str, str]] = None,
) -> list:
    """Generate product FAQs from the final enriched catalog result.

    Without *manual_knowledge*: generates 3-5 basic FAQs from the product
    data alone (title, description, tags, etc.).

    With *manual_knowledge*: generates up to 10 richer FAQs that draw from
    both the product data **and** the extracted manual content.  The prompt
    instructs the LLM to avoid duplicating what the description already
    covers, so FAQs surface genuinely new details from the manual.
    """
    has_manual = bool(manual_knowledge and any(v.strip() for v in manual_knowledge.values()))
    logger.info("[FAQ] Generating FAQs: keys=%s, locale=%s, has_manual=%s",
                list(enriched_result.keys()), locale, has_manual)

    if not (api_key := os.getenv("NGC_API_KEY")):
        raise RuntimeError(NGC_API_KEY_NOT_SET_ERROR)

    info = LOCALE_CONFIG.get(locale, {"language": "English", "region": "United States", "country": "United States", "context": "American English"})
    localized_terminology_rule = _localized_terminology_rule(info)
    localized_terminology_bullet = f"- {localized_terminology_rule}" if localized_terminology_rule else ""
    llm_config = get_config().get_llm_config()
    client = OpenAI(base_url=llm_config['url'], api_key=api_key)

    product_json = json.dumps(enriched_result, indent=2, ensure_ascii=False)

    if has_manual:
        manual_section = _format_manual_knowledge(manual_knowledge)
        prompt = f"""/no_think You are a retail product FAQ specialist. Generate up to 10 frequently asked questions and answers for the product described below. You have access to both the product listing AND extracted knowledge from the official product manual.

PRODUCT:
{product_json}

{manual_section}

TARGET LANGUAGE / REGION: {info['language']} ({info['region']})
{info['context']}

RULES:
- Generate between 5 and 10 FAQs.
- Each FAQ must have a "question" and an "answer" field.
- The product description already covers certain details. Generate FAQs about information FROM THE MANUAL that adds to or expands on the description. Do NOT create questions whose answers are fully contained in the description.
- Prioritize topics where the manual provides specific, detailed information (measurements, ratings, temperatures, durations, capacities, certifications).
- When the manual knowledge provides precise data, include those specifics in the answer.
- Answers must be helpful, concise (1-3 sentences), and factual.
- ONLY reference details present in the product data or manual knowledge above. Do NOT fabricate specifications.
- Write questions and answers in {info['language']} appropriate for {info['region']}.
{localized_terminology_bullet}

OUTPUT FORMAT:
Return ONLY a valid JSON array. No markdown, no commentary.
Example: [{{"question": "...", "answer": "..."}}, ...]"""
    else:
        prompt = f"""/no_think You are a retail product FAQ specialist. Generate 3 to 5 frequently asked questions and answers for the product described below.

PRODUCT:
{product_json}

TARGET LANGUAGE / REGION: {info['language']} ({info['region']})
{info['context']}

RULES:
- Generate between 3 and 5 FAQs.
- Each FAQ must have a "question" and an "answer" field.
- Questions should cover practical topics a shopper would ask: materials, care instructions, sizing, use cases, compatibility, durability.
- Answers must be helpful, concise (1-3 sentences), and factual.
- ONLY reference details present in the product data above. Do NOT fabricate specifications.
- Write questions and answers in {info['language']} appropriate for {info['region']}.
{localized_terminology_bullet}

OUTPUT FORMAT:
Return ONLY a valid JSON array. No markdown, no commentary.
Example: [{{"question": "...", "answer": "..."}}, ...]"""

    max_tokens = 4096 if has_manual else 2048
    logger.info("[FAQ] Sending prompt to Nemotron (length: %d chars, max_tokens: %d)", len(prompt), max_tokens)

    completion = client.chat.completions.create(
        model=llm_config['model'],
        messages=[{"role": "system", "content": "/no_think"}, {"role": "user", "content": prompt}],
        temperature=0.1, top_p=0.9, max_tokens=max_tokens, stream=True,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}}
    )

    text = "".join(
        chunk.choices[0].delta.content
        for chunk in completion
        if chunk.choices[0].delta and chunk.choices[0].delta.content
    )
    logger.info("[FAQ] Nemotron response received: %d chars", len(text))

    # Parse JSON array (inline — parse_llm_json only handles dicts)
    try:
        cleaned = text.strip()
        for marker in ("```json", "```"):
            if marker in cleaned:
                start = cleaned.find(marker) + len(marker)
                end = cleaned.find("```", start)
                if end > start:
                    cleaned = cleaned[start:end].strip()
                    break
        first_bracket = cleaned.find("[")
        last_bracket = cleaned.rfind("]")
        if first_bracket != -1 and last_bracket > first_bracket:
            cleaned = cleaned[first_bracket : last_bracket + 1]
        parsed = json.loads(cleaned)
        if isinstance(parsed, list) and all(
            isinstance(f, dict) and "question" in f and "answer" in f
            for f in parsed
        ):
            logger.info("[FAQ] Generated %d FAQs", len(parsed))
            return parsed
        logger.warning("[FAQ] Parsed JSON has unexpected structure, returning empty list")
        return []
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("[FAQ] JSON parse failed (%s), returning empty list", exc)
        return []


def _call_nemotron_extract_schema_fields(
    enriched_result: Dict[str, Any],
    locale: str = "en-US",
) -> Dict[str, Any]:
    """Extract structured product attributes from enriched data for protocol schemas.

    Uses the LLM to infer fields like brand, material, age_group, etc.
    from the product title and description. Returns a dict of extracted
    fields that can be merged into ACP/UCP schema templates.
    """
    logger.info("[Schema] Extracting structured fields for protocol schemas, locale=%s", locale)

    if not (api_key := os.getenv("NGC_API_KEY")):
        raise RuntimeError(NGC_API_KEY_NOT_SET_ERROR)

    info = LOCALE_CONFIG.get(locale, {"language": "English", "region": "United States", "country": "United States", "context": "American English"})
    llm_config = get_config().get_llm_config()
    client = OpenAI(base_url=llm_config['url'], api_key=api_key)

    product_json = json.dumps(enriched_result, indent=2, ensure_ascii=False)

    prompt = f"""/no_think You are a retail product data specialist. Analyze the product data below and extract structured attributes for commerce protocol schemas.

PRODUCT:
{product_json}

TARGET LANGUAGE / REGION: {info['language']} ({info['region']})

Extract the following fields from the product title, description, and tags. Return ONLY what can be confidently determined from the data. Use null for anything that cannot be determined.

FIELDS TO EXTRACT:
- "brand": The brand or manufacturer name (e.g., "Nature Made", "Nike", "Samsung")
- "condition": Product condition — must be one of: "new", "refurbished", "used". Default to "new" for retail products.
- "material": Primary material if mentioned (e.g., "leather", "cotton", "plastic")
- "age_group": Target age — must be one of: "newborn", "infant", "toddler", "kids", "adult". Use null if not determinable.
- "gender": Target gender — must be one of: "male", "female", "unisex". Use null if not determinable.
- "short_title": A condensed version of the title, max 65 characters
- "google_product_category": A Google product taxonomy path (e.g., "Health > Vitamins & Supplements > Fish Oil")
- "product_details": An array of key product specifications extracted from the description. Each item must have "attribute_name" and "attribute_value" fields. Extract specific, measurable attributes (quantities, weights, certifications, ratings, etc.)
- "product_highlights": An array of 3-5 concise selling points (max 150 chars each) that go beyond the tags

OUTPUT FORMAT:
Return ONLY a valid JSON object. No markdown, no commentary.
Example: {{"brand": "...", "condition": "new", "material": null, "age_group": "adult", "gender": "unisex", "short_title": "...", "google_product_category": "...", "product_details": [{{"attribute_name": "...", "attribute_value": "..."}}], "product_highlights": ["...", "..."]}}"""

    completion = client.chat.completions.create(
        model=llm_config['model'],
        messages=[{"role": "system", "content": "/no_think"}, {"role": "user", "content": prompt}],
        temperature=0.1, top_p=0.9, max_tokens=2048, stream=True,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}}
    )

    text = "".join(
        chunk.choices[0].delta.content
        for chunk in completion
        if chunk.choices[0].delta and chunk.choices[0].delta.content
    )
    logger.info("[Schema] Nemotron response received: %d chars", len(text))

    try:
        parsed = parse_llm_json(text)
        if isinstance(parsed, dict):
            logger.info("[Schema] Extracted fields: %s", list(parsed.keys()))
            return parsed
        logger.warning("[Schema] Parsed JSON is not a dict, returning empty")
        return {}
    except Exception as exc:
        logger.warning("[Schema] JSON parse failed (%s), returning empty dict", exc)
        return {}


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

    Step 1: Content enhancement + localization (conditional - only if product_data provided):
        - Merges pre-filtered product_data with VLM output
        - Applies anti-hallucination rules (no fabricated specs)
        - Localizes to target language/region
        - When no product_data, VLM output is used directly

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

    # Step 1: Only run enhancement when there is user data with actual content to merge
    has_content = filtered_product_data and any(
        v for k, v in filtered_product_data.items()
        if isinstance(v, str) and v.strip()
    )
    if has_content:
        enhanced = _call_nemotron_enhance_vlm(vlm_output, filtered_product_data, locale)
        logger.info("Step 1 complete (enhanced + localized to %s): enhanced_keys=%s", locale, list(enhanced.keys()))
    else:
        enhanced = vlm_output
        logger.info("Step 1 skipped: no product_data with content — using VLM output directly")

    # Step 2: Apply brand instructions if provided
    if brand_instructions:
        enhanced = _call_nemotron_apply_branding(enhanced, brand_instructions, locale)
        logger.info("Step 2 complete: brand-aligned content ready")
    else:
        logger.info("Step 2 skipped: no brand_instructions provided")

    if product_data and has_content:
        enhanced = _call_nemotron_resolve_merge_conflicts(vlm_output, filtered_product_data, enhanced, locale)
        logger.info("Merge QA complete: contradictions resolved")
    
    logger.info("Nemotron enhancement pipeline complete: final_keys=%s", list(enhanced.keys()))
    return enhanced

def _call_vlm(image_bytes: bytes, content_type: str, locale: str = "en-US") -> Dict[str, Any]:
    """Call VLM to analyze product image, then structure the output via LLM.

    Uses a short VLM prompt to minimize hallucinations (longer prompts degrade
    quality on this model class), then passes the free-text observation to
    _call_nemotron_structure_vlm() for JSON structuring and localization.
    """
    logger.info("Calling VLM: bytes=%d, content_type=%s, locale=%s", len(image_bytes or b""), content_type, locale)

    api_key = os.getenv("NGC_API_KEY")
    if not api_key:
        raise RuntimeError(NGC_API_KEY_NOT_SET_ERROR)

    vlm_config = get_config().get_vlm_config()
    client = OpenAI(base_url=vlm_config['url'], api_key=api_key)

    prompt_text = (
        "Describe only visible facts about this product: appearance, shape, colors, visible text, "
        "brand/labels, controls, and distinctive design. Include numbers/specs only if clearly readable "
        "as printed text; never infer capacity, size, model, power, weight, or volume."
    )

    completion = client.chat.completions.create(
        model=vlm_config['model'],
        messages=[
            {"role": "system", "content": "/no_think"},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:{content_type};base64,{base64.b64encode(image_bytes).decode()}"}},
                {"type": "text", "text": prompt_text}
            ]}
        ],
        temperature=0.1, top_p=0.9, max_tokens=4096, stream=True
    )

    text = "".join(chunk.choices[0].delta.content for chunk in completion if chunk.choices[0].delta and chunk.choices[0].delta.content)
    logger.info("VLM free-text response received: %d chars", len(text))

    return _call_nemotron_structure_vlm(text.strip(), locale)


def _call_nemotron_structure_vlm(vlm_text: str, locale: str = "en-US") -> Dict[str, Any]:
    """Structure and enhance free-text VLM output into e-commerce catalog JSON.

    Rewrites the VLM observation into polished catalog copy while staying
    faithful to the facts described. Localizes to the target language/region.
    """
    logger.info("[Structure] Structuring VLM text: %d chars, locale=%s", len(vlm_text), locale)

    if not (api_key := os.getenv("NGC_API_KEY")):
        raise RuntimeError(NGC_API_KEY_NOT_SET_ERROR)

    info = LOCALE_CONFIG.get(locale, {"language": "English", "region": "United States", "country": "United States", "context": "American English"})
    localized_terminology_block = _localized_terminology_block(info)
    llm_config = get_config().get_llm_config()
    client = OpenAI(base_url=llm_config['url'], api_key=api_key)

    categories_str = json.dumps(CATEGORY_OUTPUT_VALUES)
    colors_str = json.dumps(ALLOWED_COLORS)

    prompt = f"""/no_think Convert the visual description below into e-commerce product catalog fields. Write in polished, professional catalog language in {info['language']} for {info['region']} ({info['context']}). Do NOT invent features, materials, or specifications not mentioned in the description. Do NOT state capacity, dimensions, volume, weight, power rating, counts, compatibility, or model/spec values unless the visual description explicitly says the value appears as readable printed text. If the visual description mentions a number/spec but does not say it is readable printed text, omit it. Do NOT use size/weight claims like compact, large, spacious, lightweight, or heavy unless scale is visible in the visual description.
{localized_terminology_block}

VISUAL DESCRIPTION:
{vlm_text}

ALLOWED CATEGORIES: {categories_str}
ALLOWED COLORS: {colors_str}

RULES:
- title: Clear catalog title, not creative copy. Use only customer-facing product identity: brand/model names, official product name, product type, variant, flavor, scent, formulation, count, dosage, rating, size, material, compatibility, and model/spec values when explicitly readable or provided. Do not include packaging/container appearance such as cap color, bottle color, box color, label color, banner color, background color, shape, or label placement unless it is an official product variant or necessary retail differentiator. Do not copy visible English generic product-type label text as the localized product type; translate generic product-type words into {info['language']}. Do not include capacities, dimensions, model values, or other specs unless the visual description says they are readable printed text. Write in {info['language']}.
- description: Write as customer-facing e-commerce catalog copy in {info['language']}. Highlight the product's appeal, visible design, and visible features. Do NOT describe the label or packaging text placement (no "brand name is displayed on", "text reads", "prominently displayed", "printed in white"). Instead, naturally incorporate brand and product names into the copy.
- categories: Pick 1-2 from the allowed list. Use "uncategorized" if none fit. English.
- tags: 10 search tags derived from the text. English.
- colors: 1-2 visible product colors selected from ALLOWED COLORS only. Do not output materials, finishes, textures, or product types as colors; choose the closest visible generic color instead. English.

Return ONLY valid JSON:
{{"title": "...", "description": "...", "categories": [...], "tags": [...], "colors": [...]}}"""

    completion = client.chat.completions.create(
        model=llm_config['model'],
        messages=[{"role": "system", "content": "/no_think"}, {"role": "user", "content": prompt}],
        temperature=0.0, top_p=1, max_tokens=2048, stream=True,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}}
    )

    text = "".join(
        chunk.choices[0].delta.content
        for chunk in completion
        if chunk.choices[0].delta and chunk.choices[0].delta.content
    )
    logger.info("[Structure] LLM response received: %d chars", len(text))

    parsed = parse_llm_json(text, extract_braces=True, strip_comments=True)
    if parsed is not None:
        logger.info("[Structure] Structured successfully: keys=%s", list(parsed.keys()))
        return parsed

    logger.warning("[Structure] JSON parse failed, returning raw text as description")
    return {"title": "", "description": vlm_text, "categories": ["uncategorized"], "tags": [], "colors": []}


def extract_vlm_observation(image_bytes: bytes, content_type: str, locale: str = "en-US") -> Dict[str, Any]:
    """Run only the raw VLM observation step."""
    if not image_bytes:
        raise ValueError("image_bytes is required")
    if not isinstance(content_type, str) or not content_type.startswith("image/"):
        raise ValueError("content_type must be an image/* MIME type")

    vlm_result = _call_vlm(image_bytes, content_type, locale)
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
        _normalize_categories(enhanced.get("categories"))
        or _normalize_categories(vlm_result.get("categories"))
        or [FALLBACK_CATEGORY]
    )
    colors = _normalize_colors(enhanced.get("colors")) or _normalize_colors(vlm_result.get("colors"))

    result = {
        "title": enhanced.get("title", vlm_result.get("title", "")),
        "description": enhanced.get("description", vlm_result.get("description", "")),
        "categories": categories,
        "tags": enhanced.get("tags", vlm_result.get("tags", [])),
        "colors": colors,
    }

    if product_data:
        result["enhanced_product"] = {**product_data, **enhanced, "categories": categories, "colors": colors}

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
    vlm_result = extract_vlm_observation(image_bytes, content_type, locale)
    return build_enriched_vlm_result(vlm_result, locale, product_data, brand_instructions)
