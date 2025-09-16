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


class VLMState(TypedDict, total=False):
    image_bytes: bytes
    content_type: str
    title: str
    description: str
    categories: List[str]
    error: Optional[str]
    generated_image_b64: str
    image_path: str
    metadata_path: str
    artifact_id: str
    variation_plan: Dict[str, Any]
    flux_prompt: str


def _call_vlm(image_bytes: bytes, content_type: str) -> Dict[str, Any]:
    logger.info("Calling VLM: bytes=%d, content_type=%s", len(image_bytes or b""), content_type)
    
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError("NVIDIA_API_KEY is not set")

    vlm_config = get_config().get_vlm_config()
    client = OpenAI(base_url=vlm_config['url'], api_key=api_key)

    completion = client.chat.completions.create(
        model=vlm_config['model'],
        messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:{content_type};base64,{base64.b64encode(image_bytes).decode()}"}},
            {"type": "text", "text": """You are an image analysis assistant for a retail e-commerce site. Be descriptive and persuasive to help promote the product in the image. Create a compelling title and a concise, informative description.

Classify the product into one or more categories from this fixed allowed set only (do not invent new categories):
["clothing", "kitchen", "sports", "toys", "electronics", "furniture", "office"]
If none apply, use ["uncategorized"].

Return ONLY valid JSON with the following structure:
{
  "title": "<short, clear product name>",
  "description": "<brief but informative product description>",
  "categories": ["<one or more from the allowed set>"]
}
No extra text or commentary; only return the JSON object."""}
        ]}],
        temperature=0.9, top_p=0.9, max_tokens=1024, stream=True
    )

    text = "".join(chunk.choices[0].delta.content for chunk in completion if chunk.choices[0].delta and chunk.choices[0].delta.content)
    logger.info("VLM response received: %d chars", len(text))

    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {"title": "", "description": text, "categories": ["uncategorized"]}
    except Exception:
        return {"title": "", "description": text, "categories": ["uncategorized"]}


def _call_planner_llm(title: str, description: str, categories: List[str]) -> Dict[str, Any]:
    logger.info("Calling planner LLM: title_len=%d desc_len=%d cats=%s", len(title or ""), len(description or ""), categories)

    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError("NVIDIA_API_KEY is not set")

    llm_config = get_config().get_llm_config()
    client = OpenAI(base_url=llm_config['url'], api_key=api_key)

    completion = client.chat.completions.create(
        model=llm_config['model'],
        messages=[
            {"role": "system", "content": "/no_think You are a product image variation planner. Output ONLY valid JSON - no markdown formatting, no code blocks, no backticks. "
             "Preserve the subject identity. Change ONLY background, camera angle, lighting, and mood according to the title and description. Be creative and contextual - make backgrounds that tell a story about how/where this product would be used!"
             "Adhere to the JSON schema with fields: preserve_subject, background_style, camera_angle, lighting, color_palette, negatives, cfg_scale, steps, variants."},
            {"role": "user", "content": f"""TITLE: {title}
DESCRIPTION: {description}
CATEGORIES: {categories}

Produce ONLY a JSON object with no markdown formatting or code blocks. Example form:
{{"preserve_subject": "white ceramic mug with handle", 
"background_style": "modern kitchen counter, shallow depth of field", 
"camera_angle": "overhead", 
"lighting": "soft natural window light", 
"color_palette": "neutral, warm highlights", 
"negatives": ["do not alter the mug", "no text, no logos, no duplicates", "no changes to color, shape, handle"], 
"cfg_scale": 3.5, "steps": 30, "variants": 1}}

CRITICAL: Return the raw JSON object only - no ```json``` or ``` blocks. Keep the subject unchanged. Do not add extra keys or commentary. Use integers for steps/variants; use a float for cfg_scale between 2.5 and 4.5."""}
        ],
        temperature=0.6, top_p=1, max_tokens=1024, stream=True
    )

    text = "".join(chunk.choices[0].delta.content for chunk in completion if chunk.choices[0].delta and chunk.choices[0].delta.content).strip()
    logger.info("Planner LLM response received: %s", text)
    
    # Extract JSON from markdown code blocks if present
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
        # Handle generic code blocks
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
    
    return f"Keep {preserve} unchanged. Replace only the background with {background}. " \
           f"Use {lighting} lighting and {camera} camera angle. " \
           f"Maintain subject color, size, orientation, and material. " \
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
        plan = _call_planner_llm(state.get("title", "").strip(), state.get("description", "").strip(), state.get("categories", []))
        prompt = _render_flux_prompt(plan)
        logger.info("Planner node success: prompt_len=%d", len(prompt))
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
    image_b64 = state.get("generated_image_b64")
    if not image_b64:
        return {"error": "generated_image_b64 missing", **state}

    try:
        output_dir = os.getenv("OUTPUT_DIR", os.path.join(os.getcwd(), "data", "outputs"))
        os.makedirs(output_dir, exist_ok=True)

        artifact_id = uuid4().hex
        image_path = os.path.join(output_dir, f"{artifact_id}.png")
        metadata_path = os.path.join(output_dir, f"{artifact_id}.json")

        with open(image_path, "wb") as f:
            f.write(base64.b64decode(image_b64))

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump({
                "id": artifact_id, "title": state.get("title", ""), "description": state.get("description", ""),
                "categories": state.get("categories", []), "created_at": datetime.now(timezone.utc).isoformat(),
                "image_path": image_path, "source_content_type": state.get("content_type")
            }, f, ensure_ascii=False, indent=2)

        logger.info("Persist node success: image_path=%s metadata_path=%s", image_path, metadata_path)
        return {**state, "artifact_id": artifact_id, "image_path": image_path, "metadata_path": metadata_path}
    except Exception as exc:
        logger.exception("Persist node exception: %s", exc)
        return {"error": str(exc), **state}


def vlm_node(state: VLMState) -> VLMState:
    logger.info("VLM node start")
    image_bytes = state.get("image_bytes")
    content_type = state.get("content_type", "image/png")

    if not image_bytes:
        return {"error": "image_bytes missing or empty", **state}
    if not isinstance(content_type, str) or not content_type.startswith("image/"):
        return {"error": "content_type must be an image/* MIME type", **state}

    result = _call_vlm(image_bytes, content_type)
    logger.info("VLM node outputs: title_len=%d desc_len=%d categories=%s",
                len(result.get("title", "")), len(result.get("description", "")), result.get("categories", []))
    
    return {**state, "title": result.get("title", ""), "description": result.get("description", ""), 
            "categories": result.get("categories", ["uncategorized"])}


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
