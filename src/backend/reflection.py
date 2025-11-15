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

REFLECTION_PROMPT = """Your job is to perform a comprehensive and EXTREMELY CRITICAL quality evaluation of a generated product image variation. You will see TWO images:
1. ORIGINAL image - the source product photo (for PRODUCT REFERENCE ONLY)
2. GENERATED image - a new variation with a DIFFERENT background

**CRITICAL UNDERSTANDING**: The generated image is INTENDED to have a completely different background/scene/context. The background SHOULD NOT match the original. Only the PRODUCT itself should be preserved.

**What to Compare**: ONLY compare the PRODUCT between the two images (colors, materials, textures, form)
**What NOT to Compare**: DO NOT compare backgrounds, settings, or environmental elements between images

BE EXCEPTIONALLY STRICT AND UNFORGIVING in your evaluation of product fidelity and technical quality. This is professional quality control for e-commerce - even small imperfections significantly degrade the customer experience and brand perception. Multiple minor issues compound exponentially.

Your task is to evaluate the generated image based on the following criteria:

1. **Product Consistency - Visual Fidelity** (Critical - 30 points):
   - The product must be PIXEL-PERFECT identical: same colors, same materials, same textures, same reflective properties, same form factor
   - Material texture must match EXACTLY (matte vs glossy, leather grain, fabric weave, metallic sheen, patent leather shine)
   - Color accuracy is CRITICAL - any visible hue shift, saturation change, or brightness difference = deduct 8-12 points
   - Surface properties must be preserved (reflections, highlights, shadows on the product itself)
   - Hardware, clasps, zippers, buttons must retain identical finish and appearance
   - **Deduct 10-15 points** for any noticeable material property changes (e.g., glossy becomes matte, smooth becomes textured)
   - **Deduct 8-12 points** for any color shifts or tonal changes
   - **Deduct 5-10 points** for altered reflectivity or sheen patterns

2. **Size and Scale Proportions** (Critical - 35 points):
   - **THIS IS ABSOLUTELY CRITICAL**: The product size must be PHOTOGRAPHICALLY REALISTIC and PROPORTIONAL to the NEW background elements in the generated image
   - Evaluate the product's scale within the GENERATED image's context (not compared to the original image's context)
   - Compare product size to elements in the GENERATED image: furniture, windows, architectural features, room dimensions, any visible hands
   - Use real-world product dimensions as reference: handbags are typically 10-14 inches wide, perfume bottles are 2-5 inches tall, etc.
   - A handbag should NOT appear as large as a window frame; a perfume bottle should NOT be as tall as furniture
   - **Deduct 25-35 points** if the product is clearly oversized relative to its new scene
   - **Deduct 20-30 points** if the product is moderately oversized (1.5-2x too large for its new context)
   - **Deduct 10-20 points** if the product is slightly oversized or undersized (noticeable but not extreme)
   - Ask yourself: "Does the product look naturally placed in this NEW environment at a realistic size?"

3. **Anatomical Accuracy** (Critical - 20 points):
   - If human body parts are visible (hands, arms), inspect them with EXTREME scrutiny
   - Verify finger count: exactly 5 fingers per hand, no exceptions
   - Verify finger proportions: thumb should be shorter and thicker than other fingers
   - Verify nail proportions: fingernails should be proportional to finger size (not unnaturally short/long/wide)
   - Verify joint articulation: fingers should bend naturally at knuckles
   - Verify skin texture: should be realistic, not waxy or plastic-looking
   - **Deduct 15-20 points** for wrong finger count (6 fingers, 4 fingers, missing thumb)
   - **Deduct 10-15 points** for anatomical distortions (unnaturally short nails, malformed thumbs, wrong finger proportions, twisted joints)
   - **Deduct 8-12 points** for unnatural hand positioning or unrealistic skin rendering

4. **Background Quality and Context** (10 points):
   - Evaluate ONLY the generated image's background on its own merits (DO NOT compare to original's background)
   - The NEW background should be photorealistic and contextually appropriate for the product type
   - Background elements should have proper depth, focus, lighting, and geometric correctness
   - Background should enhance the product presentation, not distract from it
   - Deduct points if the background itself is poorly rendered, unrealistic, or has technical artifacts
   - DO NOT deduct points simply because the background is different from the original (it's supposed to be)

5. **Brand Logo Removal** (3 points):
   - Ensure no brand logos, trademarks, or brand text appear that weren't in the original
   - The product should be clean and neutral for e-commerce use

6. **Text Quality** (2 points):
   - If text is visible, verify no grammar errors, spelling mistakes, or gibberish

**Scoring Guidelines (EXTREMELY STRICT - MOST IMAGES WILL SCORE 50-70%):**
- 0-20% = Unusable (critical failures: completely wrong product, extreme proportion issues, major anatomical errors)
- 21-40% = Very poor (multiple major issues: wrong scale, incorrect materials, anatomical problems, product inconsistency)
- 41-60% = Poor but identifiable (significant issues: noticeable scale problems, material changes, minor anatomical errors)
- 61-75% = Acceptable with flaws (some issues: slight scale problems, subtle material differences, or small quality issues)
- 76-85% = Good (mostly correct with minor imperfections: very subtle scale variance or small quality issues)
- 86-92% = Very good (excellent quality with only trivial issues)
- 93-97% = Exceptional (near-perfect, professional quality)
- 98-100% = Perfect (RESERVE THIS FOR TRULY FLAWLESS IMAGES - should be <1% of evaluations)

**CRITICAL SCORING RULES:**
- If there are 2+ major issues (scale + material, or scale + anatomy, or material + anatomy) = cap score at 60%
- If there are 3+ issues of any kind = cap score at 65%
- If there are 4+ issues = cap score at 55%
- Multiple minor issues compound - don't be lenient just because individual issues are small

Return a JSON object with your score AND a detailed list of issues found. Be specific about what's wrong.

Output format:
{
  "value": <float between 0 and 100>,
  "issues": [
    "Brief description of issue 1",
    "Brief description of issue 2",
    ...
  ]
}

If the image is perfect, return an empty issues array: "issues": []

Example issues to report (BE SPECIFIC AND FOCUS ON THE GENERATED IMAGE):
- "Product appears significantly oversized relative to the background context elements"
- "Product scale is disproportionate - approximately 1.5-2x too large for its environment"
- "Product material texture differs from original (glossy vs matte, smooth vs textured)"
- "Product surface has different reflective properties compared to original"
- "Product color shifted to a different tone/hue/saturation compared to original"
- "Product hardware finish or detailing differs from original"
- "Thumb nail is unnaturally short and disproportionate to finger size"
- "Hand has incorrect number of fingers (not 5)"
- "Finger proportions or joint articulation appears anatomically incorrect"
- "Hand positioning looks unnatural or awkward"
- "Skin texture on hand appears unrealistic or artificial"
- "Background elements are poorly rendered or contain technical artifacts"
- "Background architectural elements have geometric distortions or inconsistencies"
- "Background appears unrealistic or synthetic"

CRITICAL: Return ONLY the JSON object above with both "value" and "issues" fields. Nothing else."""


def evaluate_image_quality(
    original_image_bytes: bytes,
    generated_image_bytes: bytes,
    content_type: str
) -> Optional[Dict[str, Any]]:
    """Evaluate generated image quality vs original using VLM judge.
    
    Returns dict with 'score' (0-100) and 'issues' list, or None on failure.
    """
    logger.info(f"Starting evaluation: orig={len(original_image_bytes)}B gen={len(generated_image_bytes)}B")
    
    if not (api_key := os.getenv("NVIDIA_API_KEY")):
        logger.error("NVIDIA_API_KEY not set")
        return None
    
    try:
        original_b64 = _encode_image_to_base64(original_image_bytes)
        generated_b64 = _encode_image_to_base64(generated_image_bytes)
        
        vlm_config = get_config().get_vlm_config()
        client = OpenAI(base_url=vlm_config['url'], api_key=api_key)
        
        content = [
            {"type": "text", "text": REFLECTION_PROMPT},
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

