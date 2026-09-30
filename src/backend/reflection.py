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

"""Image quality evaluation for generated variations using VLM."""
import os
import base64
import logging
from typing import Optional, Dict, Any
from io import BytesIO
from PIL import Image

from dotenv import load_dotenv
from openai import OpenAI
from backend.config import get_config
from backend.utils import parse_llm_json

load_dotenv()

logger = logging.getLogger("catalog_enrichment.reflection")

REFLECTION_PROMPT_TEMPLATE = """You are a strict product-image QA judge. You will receive exactly two images:
IMAGE 1: ORIGINAL REFERENCE product image.
IMAGE 2: GENERATED VARIATION to judge against Image 1.

First verify that you can inspect both images. If either image is missing, unavailable, duplicated incorrectly, or impossible to compare, return score 0 and include a specific issue explaining which input is missing or unusable.

EXPECTED GENERATION INTENT:
{generation_prompt_section}

Image 2 is expected to be a variation with a different background, setting, camera angle, lighting, mood, or staging when requested by the generation intent. Do not list expected contextual/background differences as quality issues, and do not lower the score for them by themselves. Penalize the background or staging if it corrupts, hides, occludes, distorts, duplicates, visually blends into, physically misplaces, or otherwise damages the product.

Do not assume the product was preserved just because the generation intent says to keep it unchanged. The intent only describes what was requested; judge ONLY what is actually visible in the images.

Before scoring, transcribe the visible text on the product in each image:
- original_text: every readable word, number, logo text, and symbol on the product in Image 1, exactly as written (preserve spelling and case).
- generated_text: the same for the product in Image 2, exactly as written.
Compare the two lists character by character. Any missing, added, misspelled, garbled, or changed text (including numbers, sizes, and units) is a product-fidelity defect that must be listed as an issue and must cap the score at 84 or lower. If the product has no readable text, use empty lists.

Compare the {product_name} in Image 2 against the same product in Image 1. Judge product fidelity first:
- Product presence: the same product must be clearly visible in Image 2.
- Shape and structure: silhouette, components, handles, straps, caps, pockets, buttons, controls, ports, labels, and other visible parts must match.
- Color and material: hue, texture, finish, transparency, reflectiveness, and material cues must match.
- Text and markings: readable labels, logos, numbers, and symbols must not be removed, invented, garbled, or moved in a misleading way.
- Proportion and scale: the product must remain realistic and not stretched, warped, oversized, undersized, cropped, or partly hidden.
- Functional realism: the setting, support surface, placement, clearance, indoor/outdoor context, and safety implications must make real-world sense for the product's apparent function, scale, weight, mobility, heat, airflow, power/fuel/water needs, cleanliness, and typical use.
- Background quality: the background may differ, but it must not corrupt, occlude, distort, duplicate, visually blend into, physically misplace, or make the product scenario impossible or unsafe.
- People/body parts: if present, hands, fingers, faces, or bodies must look natural and must not alter the product.

Issue reporting rules:
- List every clearly visible mismatch as a concise issue.
- If the score is below 100, issues must be non-empty.
- Always include a concise rationale explaining the score. If the score is 100, explain why the product fidelity is perfect despite any intended background/context change.
- Do not use a default high score. Score 95 or higher only when product identity, structure, colors, materials, text/labels, and proportions are nearly perfect.

Scoring rubric:
- 95-100: Near-perfect product fidelity with no or only tiny visible differences.
- 85-94: Minor product differences that most shoppers may not notice.
- 70-84: Noticeable differences or mild product distortion.
- 50-69: Major product differences, obvious distortion, scale problems, or missing details.
- 1-49: Product heavily altered, partially missing, badly corrupted, or hard to recognize.
- 0: Product missing/replaced, or both images cannot be inspected and compared.

Return ONLY JSON:
{{"original_text": ["..."], "generated_text": ["..."], "value": <float>, "rationale": "concise explanation of the score", "issues": ["issue1", "issue2", ...]}}"""


def _format_generation_prompt_section(generation_prompt: Optional[str]) -> str:
    """Format the prompt that produced Image 2 for the reflection judge."""
    if isinstance(generation_prompt, str) and generation_prompt.strip():
        return (
            "Image 2 was generated with this image-edit prompt. Use it only to identify intended "
            "background/context changes; it does not guarantee the product was preserved:\n"
            f"{generation_prompt.strip()}"
        )
    return (
        "No image-edit prompt was provided. Assume Image 2 may intentionally have a different "
        "background/context, but still judge whether the product itself stayed faithful to Image 1."
    )


def _build_reflection_messages(
    original_image_b64: str,
    generated_image_b64: str,
    reflection_prompt: str,
) -> list[Dict[str, Any]]:
    """Build an explicit two-image VLM request for the reflection judge."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "IMAGE 1 - ORIGINAL REFERENCE product image:"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{original_image_b64}"}},
                {"type": "text", "text": "IMAGE 2 - GENERATED VARIATION to judge against Image 1:"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{generated_image_b64}"}},
                {"type": "text", "text": reflection_prompt},
            ],
        }
    ]


def _count_message_images(messages: list[Dict[str, Any]]) -> int:
    """Count image_url content blocks in a chat-completion message payload."""
    count = 0
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        count += sum(1 for part in content if isinstance(part, dict) and part.get("type") == "image_url")
    return count


def evaluate_image_quality(
    original_image_bytes: bytes,
    generated_image_bytes: bytes,
    content_type: str,
    product_title: Optional[str] = None,
    generation_prompt: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Evaluate generated image quality vs original using VLM judge.
    
    Args:
        original_image_bytes: Original product image bytes
        generated_image_bytes: Generated variation image bytes
        content_type: Image content type
        product_title: Product name/title to focus the VLM evaluation
        generation_prompt: Prompt used to generate the variation image
    
    Returns:
        dict with 'score' (0-100), 'rationale', and 'issues' list, or None on failure.
    """
    product_name = product_title if product_title else "product"
    
    logger.info(f"Starting evaluation for '{product_name}': orig={len(original_image_bytes)}B gen={len(generated_image_bytes)}B")
    
    if not (api_key := os.getenv("NGC_API_KEY")):
        logger.error("NGC_API_KEY not set")
        return None
    
    try:
        original_b64 = _encode_image_to_base64(original_image_bytes)
        generated_b64 = _encode_image_to_base64(generated_image_bytes)
        
        vlm_config = get_config().get_vlm_config()
        client = OpenAI(base_url=vlm_config['url'], api_key=api_key)
        
        reflection_prompt = REFLECTION_PROMPT_TEMPLATE.format(
            product_name=product_name,
            generation_prompt_section=_format_generation_prompt_section(generation_prompt),
        )
        
        messages = _build_reflection_messages(original_b64, generated_b64, reflection_prompt)
        image_input_count = _count_message_images(messages)
        if image_input_count != 2:
            raise RuntimeError(f"Reflection VLM payload must include exactly 2 images, got {image_input_count}")

        logger.info("Calling VLM judge: model=%s image_inputs=%d", vlm_config["model"], image_input_count)
        completion = client.chat.completions.create(
            model=vlm_config['model'],
            messages=messages,
            temperature=0.0,
            top_p=0.9,
            max_tokens=8192,
            stream=False,
            # Thinking stays enabled for the judge: without it the model echoes the
            # generation prompt and scores 100 without actually comparing the images.
            extra_body={"chat_template_kwargs": {"enable_thinking": True}},
        )
        
        response_text = (completion.choices[0].message.content or "").strip()
        logger.info(f"VLM response: {response_text}")
        
        result = _parse_quality_response(response_text)
        
        if result:
            logger.info(f"Evaluation complete: score={result['score']:.1f} issues={len(result['issues'])}")
            if result['issues']:
                logger.info(f"Issues: {result['issues']}")
        else:
            logger.warning("Failed to parse response")
        
        return result
        
    except Exception as exc:
        logger.exception(f"Evaluation failed: {exc}")
        return None


def _encode_image_to_base64(image_bytes: bytes, target_format: str = "png") -> str:
    """Encode image bytes to base64, converting to target format."""
    try:
        img = Image.open(BytesIO(image_bytes))
        
        if target_format.lower() in ("jpeg", "jpg") and img.mode in ("RGBA", "P"):
            rgb = Image.new("RGB", img.size, (255, 255, 255))
            rgb.paste(img, mask=img.split()[3] if img.mode == "RGBA" else None)
            img = rgb
        
        buf = BytesIO()
        img.save(buf, format=target_format.upper())
        return base64.b64encode(buf.getvalue()).decode("utf-8")
        
    except Exception as e:
        logger.warning(f"Image conversion failed, using raw: {e}")
        return base64.b64encode(image_bytes).decode("utf-8")


def _parse_quality_response(response_text: str) -> Optional[Dict[str, Any]]:
    """Parse VLM quality response, handling JSON or markdown-wrapped JSON."""
    if "</think>" in response_text:
        response_text = response_text.split("</think>", 1)[1]
    data = parse_llm_json(response_text)
    if data is None:
        logger.warning(f"Parse failed - Response: {response_text}")
        return None

    if "value" not in data:
        logger.warning(f"Response missing 'value': {data}")
        return None

    try:
        score = max(0.0, min(100.0, float(data["value"])))
    except (ValueError, TypeError) as e:
        logger.warning(f"Invalid 'value': {e} - Response: {response_text}")
        return None

    issues = data.get("issues", []) if isinstance(data.get("issues"), list) else []
    rationale = data.get("rationale")
    if not isinstance(rationale, str):
        rationale = ""
    return {"score": score, "rationale": rationale.strip(), "issues": issues}
