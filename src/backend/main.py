import os
import json
import base64
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse


load_dotenv()


app: Any = FastAPI()


@app.get("/")
async def homepage() -> PlainTextResponse:
    return PlainTextResponse("Catalog Enrichment Backend")


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})


@app.post("/vlm/describe")
async def vlm_describe(image: UploadFile = File(...)) -> JSONResponse:
    try:
        image_bytes = await image.read()
        if not image_bytes:
            return JSONResponse({"detail": "Uploaded file is empty"}, status_code=400)

        content_type = getattr(image, "content_type", None) or "image/png"
        if not content_type.startswith("image/"):
            return JSONResponse({"detail": "File must be an image"}, status_code=400)

        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        api_key = os.getenv("NVIDIA_API_KEY")
        base_url = os.getenv("NVIDIA_API_BASE_URL", "https://integrate.api.nvidia.com/v1")

        if not api_key:
            return JSONResponse({"detail": "NVIDIA_API_KEY is not set"}, status_code=500)

        client = OpenAI(base_url=base_url, api_key=api_key)

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
                        {"type": "text", "text": """You are an image analysis assistant for a retail e-commerce site. Be descriptive and persuasive to help promote the product in the image. Create a compelling title and a concise, informative description.

Classify the product into one or more categories from this fixed allowed set only (do not invent new categories):
["clothing", "kitchen", "sports", "toys", "electronics"]
If none apply, use ["uncategorized"].

Return ONLY valid JSON with the following structure:
{
  "title": "<short, clear product name>",
  "description": "<brief but informative product description>",
  "categories": ["<one or more from the allowed set>"]
}
No extra text or commentary; only return the JSON object."""},
                    ],
                }
            ],
            temperature=0.9,
            top_p=0.9,
            max_tokens=1024,
            stream=True,
        )

        description_chunks: list[str] = []
        for chunk in completion:
            delta = chunk.choices[0].delta
            if delta and delta.content is not None:
                description_chunks.append(delta.content)

        description_text = "".join(description_chunks).strip()

        # Try to return the model output as proper JSON if possible
        try:
            parsed = json.loads(description_text)
            if isinstance(parsed, dict):
                return JSONResponse(parsed)
        except Exception:
            pass

        # Fallback: wrap the text into the expected structure
        return JSONResponse({
            "title": "",
            "description": description_text,
        })

    except Exception as exc:
        return JSONResponse({"detail": str(exc)}, status_code=500)


