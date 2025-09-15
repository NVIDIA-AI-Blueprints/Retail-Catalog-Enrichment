import os
import json
import base64
from typing import TypedDict, Optional, List, Dict, Any

from dotenv import load_dotenv
from openai import OpenAI
from langgraph.graph import StateGraph


load_dotenv()


class VLMState(TypedDict, total=False):
    """State passed through the LangGraph for VLM description.

    Required inputs:
    - image_bytes: Raw bytes of the uploaded image
    - content_type: MIME type (e.g., "image/png")

    Produced outputs:
    - title: Product title
    - description: Product description
    - categories: List of allowed categories
    - error: Error message if any
    """

    image_bytes: bytes
    content_type: str

    title: str
    description: str
    categories: List[str]
    error: Optional[str]


def _call_vlm(image_bytes: bytes, content_type: str) -> Dict[str, Any]:
    api_key = os.getenv("NVIDIA_API_KEY")
    base_url = os.getenv("NVIDIA_API_BASE_URL", "https://integrate.api.nvidia.com/v1")

    if not api_key:
        raise RuntimeError("NVIDIA_API_KEY is not set")

    client = OpenAI(base_url=base_url, api_key=api_key)

    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    completion = client.chat.completions.create(
        model="nvidia/llama-3.1-nemotron-nano-vl-8b-v1",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{content_type};base64,{image_b64}"},
                    },
                    {
                        "type": "text",
                        "text": """You are an image analysis assistant for a retail e-commerce site. Be descriptive and persuasive to help promote the product in the image. Create a compelling title and a concise, informative description.

Classify the product into one or more categories from this fixed allowed set only (do not invent new categories):
["clothing", "kitchen", "sports", "toys", "electronics"]
If none apply, use ["uncategorized"].

Return ONLY valid JSON with the following structure:
{
  "title": "<short, clear product name>",
  "description": "<brief but informative product description>",
  "categories": ["<one or more from the allowed set>"]
}
No extra text or commentary; only return the JSON object.""",
                    },
                ],
            }
        ],
        temperature=0.9,
        top_p=0.9,
        max_tokens=1024,
        stream=True,
    )

    chunks: list[str] = []
    for chunk in completion:
        delta = chunk.choices[0].delta
        if delta and delta.content is not None:
            chunks.append(delta.content)

    text = "".join(chunks).strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    # Fallback minimal structure
    return {
        "title": "",
        "description": text,
        "categories": ["uncategorized"],
    }


def vlm_node(state: VLMState) -> VLMState:
    """LangGraph node that performs VLM describe from image bytes.

    Expects 'image_bytes' and 'content_type' in the state.
    """
    image_bytes = state.get("image_bytes")
    content_type = state.get("content_type", "image/png")

    if not image_bytes:
        return {"error": "image_bytes missing or empty", **state}

    if not isinstance(content_type, str) or not content_type.startswith("image/"):
        return {"error": "content_type must be an image/* MIME type", **state}

    result = _call_vlm(image_bytes=image_bytes, content_type=content_type)

    return {
        **state,
        "title": result.get("title", ""),
        "description": result.get("description", ""),
        "categories": result.get("categories", ["uncategorized"]),
    }


def create_compiled_graph():
    """Create and compile the minimal LangGraph with a single VLM node.

    Entry and finish points are both the VLM node for now.
    """
    graph = StateGraph(VLMState)
    graph.add_node("vlm", vlm_node)
    graph.set_entry_point("vlm")
    graph.set_finish_point("vlm")
    return graph.compile()


