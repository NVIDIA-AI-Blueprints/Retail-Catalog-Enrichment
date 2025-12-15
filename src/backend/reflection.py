# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import json
import base64
import logging
from typing import Optional, Dict, Any
from io import BytesIO
from PIL import Image

from dotenv import load_dotenv
from openai import OpenAI
from backend.config import get_config

load_dotenv()

logger = logging.getLogger("catalog_enrichment.reflection")

REFLECTION_PROMPT_TEMPLATE = """Evaluate the quality of a generated product variation. You will see TWO images:
1. ORIGINAL - source product photo (REFERENCE ONLY for the {product_name})
2. GENERATED - new variation with DIFFERENT background (backgrounds SHOULD differ)

**FOCUS**: Compare ONLY the {product_name} itself between images. Ignore background differences.

**Evaluation Criteria** (100 points total, be EXTREMELY STRICT):

1. **Product Structure & Form Fidelity** (35 points):
   - The {product_name} must be structurally identical: same shape, components, and features
   - CRITICAL: Check if product has same structural elements
   - Verify no features added or removed
   - Colors, materials, textures, and reflective properties must match EXACTLY
   - Hardware/details must retain identical finish
   - Deduct 15-25 points for structural changes
   - Deduct 10-15 points for material property changes (texture differences)
   - Deduct 8-12 points for color/tone shifts

2. **Size & Scale Proportions** (35 points):
   - Product size must be REALISTIC relative to GENERATED image's context (furniture, architecture, hands)
   - Deduct 25-35 points if clearly oversized/undersized
   - Deduct 15-25 points if moderately disproportionate
   - Deduct 10-20 points if slightly off-scale

3. **Anatomical Accuracy** (20 points) - IF hands/body parts visible:
   - Exactly 5 fingers per hand with correct proportions
   - Natural joint articulation and realistic skin texture
   - Deduct 15-20 points for wrong finger count or major distortions
   - Deduct 10-15 points for anatomical abnormalities (malformed thumbs, wrong nail proportions)

4. **Background Quality** (10 points):
   - Evaluate ONLY generated background's photorealism and technical quality
   - DO NOT penalize simply because background differs from original (it should differ)
   - Deduct for poor rendering, geometric distortions, or artifacts

**Scoring Scale** (STRICT - most images score 50-70%):
- 0-40%: Unusable (critical failures)
- 41-60%: Poor (significant issues)
- 61-75%: Acceptable with flaws
- 76-85%: Good (minor imperfections)
- 86-95%: Very good
- 96-100%: Exceptional (rare, <1%)

**Compounding Rule**: 2+ major issues → cap at 60%; 3+ issues → cap at 65%; 4+ issues → cap at 55%

Return ONLY this JSON:
{{
  "value": <float 0-100>,
  "issues": ["specific issue 1", "specific issue 2", ...]
}}

Empty issues array if perfect. Focus on the {product_name} specifically."""


def evaluate_image_quality(
    original_image_bytes: bytes,
    generated_image_bytes: bytes,
    content_type: str,
    product_title: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """Evaluate generated image quality vs original using VLM judge.
    
    Args:
        original_image_bytes: Original product image bytes
        generated_image_bytes: Generated variation image bytes
        content_type: Image content type
        product_title: Product name/title to focus the VLM evaluation
    
    Returns:
        dict with 'score' (0-100) and 'issues' list, or None on failure.
    """
    product_name = product_title if product_title else "product"
    
    logger.info(f"Starting evaluation for '{product_name}': orig={len(original_image_bytes)}B gen={len(generated_image_bytes)}B")
    
    if not (api_key := os.getenv("NVIDIA_API_KEY")):
        logger.error("NVIDIA_API_KEY not set")
        return None
    
    try:
        original_b64 = _encode_image_to_base64(original_image_bytes)
        generated_b64 = _encode_image_to_base64(generated_image_bytes)
        
        vlm_config = get_config().get_vlm_config()
        client = OpenAI(base_url=vlm_config['url'], api_key=api_key)
        
        reflection_prompt = REFLECTION_PROMPT_TEMPLATE.format(product_name=product_name)
        
        content = [
            {"type": "text", "text": reflection_prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{original_b64}"}},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{generated_b64}"}}
        ]
        
        logger.info(f"Calling VLM: model={vlm_config['model']}")
        completion = client.chat.completions.create(
            model=vlm_config['model'],
            messages=[
                {"role": "system", "content": "/no_think"},
                {"role": "user", "content": content}
            ],
            temperature=0.1,
            top_p=0.9,
            max_tokens=768,
            stream=False
        )
        
        response_text = completion.choices[0].message.content.strip()
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
    try:
        text = response_text.strip()
        
        if "```" in text:
            start = text.find("```json") + 7 if "```json" in text else text.find("```") + 3
            end = text.find("```", start)
            text = text[start:end].strip() if end > start else text
        
        data = json.loads(text)
        
        if isinstance(data, dict) and "value" in data:
            score = max(0.0, min(100.0, float(data["value"])))
            issues = data.get("issues", []) if isinstance(data.get("issues"), list) else []
            return {"score": score, "issues": issues}
        
        logger.warning(f"Response missing 'value': {data}")
        return None
        
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        logger.warning(f"Parse failed: {e} - Response: {response_text}")
        return None

