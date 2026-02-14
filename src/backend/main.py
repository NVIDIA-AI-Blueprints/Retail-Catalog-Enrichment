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

import asyncio
import base64
import json
import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, Response
import httpx
from openai import APIConnectionError

from backend.vlm import run_vlm_analysis
from backend.image import generate_image_variation
from backend.trellis import generate_3d_asset
from backend.config import get_config

load_dotenv()

logger = logging.getLogger("catalog_enrichment.api")
VALID_LOCALES = {"en-US", "en-GB", "en-AU", "en-CA", "es-ES", "es-MX", "es-AR", "es-CO", "fr-FR", "fr-CA"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logger.info("App startup complete")
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://frontend:3000",
        "http://catalog-enrichment-frontend:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def homepage() -> PlainTextResponse:
    logger.info("GET /")
    return PlainTextResponse("Catalog Enrichment Backend")

@app.get("/health")
async def health() -> JSONResponse:
    logger.info("GET /health")
    return JSONResponse({"status": "ok"})

@app.get("/health/nims")
async def health_nims() -> JSONResponse:
    """
    Check the health status of all NVIDIA NIM endpoints.
    
    Returns the health status of VLM, LLM, FLUX, and TRELLIS services.
    Each service is checked by calling its /v1/health/ready endpoint.
    """
    logger.debug("GET /health/nims - checking all NIM endpoints")
    config = get_config()
    
    async def check_service(name: str, base_url: str) -> str:
        """Check if a service is healthy by calling its health endpoint."""
        health_base = base_url.rstrip('/').removesuffix('/infer')
        health_url = f"{health_base}/health/ready"
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                response = await client.get(health_url)
                if response.status_code == 200:
                    data = response.json()
                    # Check for VLM/LLM format: {"object":"health.response","message":"Service is ready."} or {"object":"health.response","status":"ok"}
                    if data.get("object") == "health.response":
                        msg = (data.get("message") or "").lower().rstrip(".")
                        if msg == "service is ready" or data.get("status") == "ok":
                            logger.debug(f"{name} service is healthy (VLM/LLM format)")
                            return "healthy"
                    # Check for FLUX/TRELLIS format: {"description":"Triton readiness check","status":"ready"}
                    if data.get("status") == "ready":
                        logger.debug(f"{name} service is healthy (Triton format)")
                        return "healthy"
                logger.warning(f"{name} service returned unexpected response: status={response.status_code}, data={data}")
                return "unhealthy"
        except Exception as e:
            logger.warning(f"{name} service health check failed: {e}")
            return "unhealthy"
    
    # Get all NIM configurations
    try:
        vlm_config = config.get_vlm_config()
        llm_config = config.get_llm_config()
        flux_config = config.get_flux_config()
        trellis_config = config.get_trellis_config()
        
        # Check all services concurrently
        vlm_status, llm_status, flux_status, trellis_status = await asyncio.gather(
            check_service("VLM", vlm_config["url"]),
            check_service("LLM", llm_config["url"]),
            check_service("FLUX", flux_config["url"]),
            check_service("TRELLIS", trellis_config["url"])
        )
        
        result = {
            "vlm": vlm_status,
            "llm": llm_status,
            "flux": flux_status,
            "trellis": trellis_status
        }
        
        logger.debug(f"NIM health check results: {result}")
        return JSONResponse(result)
        
    except Exception as e:
        logger.error(f"Error checking NIM health: {e}")
        return JSONResponse({
            "vlm": "unhealthy",
            "llm": "unhealthy",
            "flux": "unhealthy",
            "trellis": "unhealthy"
        })

@app.post("/vlm/analyze")
async def vlm_analyze(
    image: UploadFile = File(...),
    locale: str = Form("en-US"),
    product_data: str = Form(None),
    brand_instructions: str = Form(None)
) -> JSONResponse:
    """
    Fast endpoint: Analyze image and extract product fields using VLM.
    
    This endpoint runs ONLY the VLM analysis (no image generation).
    Returns fields quickly (~2-5 seconds).
    """
    try:
        if locale not in VALID_LOCALES:
            logger.error(f"/vlm/analyze error: invalid locale={locale}")
            return JSONResponse({"detail": f"Invalid locale. Supported locales: {sorted(VALID_LOCALES)}"}, status_code=400)
        
        product_json = None
        if product_data:
            try:
                product_json = json.loads(product_data)
                logger.info(f"Parsed product_data: {product_json}")
            except Exception as e:
                logger.error(f"/vlm/analyze error: invalid JSON in product_data: {e}")
                return JSONResponse({"detail": f"Invalid JSON in product_data: {e}"}, status_code=400)
        
        validation_result, error_response = await _validate_image(image, "/vlm/analyze")
        if error_response:
            return error_response
        image_bytes, content_type = validation_result
        
        logger.info(f"Running VLM analysis: locale={locale} mode={'augmentation' if product_json else 'generation'}")
        result = run_vlm_analysis(image_bytes, content_type, locale, product_json, brand_instructions)
        
        payload = {
            "title": result.get("title", ""),
            "description": result.get("description", ""),
            "categories": result.get("categories", ["uncategorized"]),
            "tags": result.get("tags", []),
            "colors": result.get("colors", []),
            "locale": locale
        }
        
        if result.get("enhanced_product"):
            payload["enhanced_product"] = result["enhanced_product"]
        
        logger.info(f"/vlm/analyze success: title_len={len(payload['title'])} desc_len={len(payload['description'])} locale={locale}")
        return JSONResponse(payload)
        
    except (APIConnectionError, httpx.ConnectError) as exc:
        logger.exception(f"/vlm/analyze connection error: {exc}")
        return JSONResponse({
            "detail": "Unable to connect to the NIM endpoint. Please verify that the NVIDIA NIM container is running."
        }, status_code=503)
    except Exception as exc:
        logger.exception(f"/vlm/analyze exception: {exc}")
        return JSONResponse({"detail": str(exc)}, status_code=500)


@app.post("/generate/variation")
async def generate_variation(
    image: UploadFile = File(...),
    locale: str = Form("en-US"),
    title: str = Form(...),
    description: str = Form(...),
    categories: str = Form(...),
    tags: str = Form("[]"),
    colors: str = Form("[]"),
    enhanced_product: str = Form(None)
) -> JSONResponse:
    """
    Slow endpoint: Generate image variation given VLM analysis results.
    
    Takes pre-computed fields from /vlm/analyze and generates a new image variation.
    Returns generated image (~30-60 seconds).
    """
    try:
        if locale not in VALID_LOCALES:
            logger.error(f"/generate/variation error: invalid locale={locale}")
            return JSONResponse({"detail": f"Invalid locale. Supported locales: {sorted(VALID_LOCALES)}"}, status_code=400)
        
        # Parse JSON fields
        try:
            categories_list = json.loads(categories)
            tags_list = json.loads(tags)
            colors_list = json.loads(colors)
            enhanced_product_dict = json.loads(enhanced_product) if enhanced_product else None
        except Exception as e:
            logger.error(f"/generate/variation error: invalid JSON in fields: {e}")
            return JSONResponse({"detail": f"Invalid JSON in fields: {e}"}, status_code=400)
        
        validation_result, error_response = await _validate_image(image, "/generate/variation")
        if error_response:
            return error_response
        image_bytes, content_type = validation_result
        
        logger.info(f"Generating image variation: title_len={len(title)} locale={locale}")
        result = await generate_image_variation(
            image_bytes=image_bytes,
            content_type=content_type,
            title=title,
            description=description,
            categories=categories_list,
            tags=tags_list,
            colors=colors_list,
            locale=locale,
            enhanced_product=enhanced_product_dict
        )
        
        payload = {
            "generated_image_b64": result["generated_image_b64"],
            "artifact_id": result["artifact_id"],
            "image_path": result["image_path"],
            "metadata_path": result["metadata_path"],
            "quality_score": result["quality_score"],
            "quality_issues": result["quality_issues"],
            "locale": locale
        }
        
        logger.info(f"/generate/variation success: artifact_id={result['artifact_id']} image_b64_len={len(result['generated_image_b64'])} quality_score={result['quality_score']} issues_count={len(result['quality_issues'])}")
        return JSONResponse(payload)
        
    except (APIConnectionError, httpx.ConnectError) as exc:
        logger.exception(f"/generate/variation connection error: {exc}")
        return JSONResponse({
            "detail": "Unable to connect to the NIM endpoint. Please verify that the NVIDIA FluxNIM container is running."
        }, status_code=503)
    except Exception as exc:
        logger.exception(f"/generate/variation exception: {exc}")
        return JSONResponse({"detail": str(exc)}, status_code=500)


async def _validate_image(image: UploadFile, endpoint: str):
    logger.info(f"POST {endpoint} filename={getattr(image, 'filename', None)} content_type={getattr(image, 'content_type', None)}")
    image_bytes = await image.read()
    
    if not image_bytes:
        logger.error(f"{endpoint} error: empty upload")
        return None, JSONResponse({"detail": "Uploaded file is empty"}, status_code=400)
    
    content_type = getattr(image, "content_type", None) or "image/png"
    if not content_type.startswith("image/"):
        logger.error(f"{endpoint} error: non-image content_type={content_type}")
        return None, JSONResponse({"detail": "File must be an image"}, status_code=400)
    
    return (image_bytes, content_type), None


@app.post("/generate/3d")
async def generate_3d(
    image: UploadFile = File(...),
    slat_cfg_scale: float = Form(5.0),
    ss_cfg_scale: float = Form(10.0),
    slat_sampling_steps: int = Form(50),
    ss_sampling_steps: int = Form(50),
    seed: int = Form(0),
    return_json: bool = Form(False)
) -> Response:
    """
    Generate a 3D GLB asset from a 2D product image using TRELLIS model.
    
    This endpoint accepts a product image and returns a 3D GLB file that can be rendered in the UI.
    Processing time: ~30-120 seconds depending on parameters.
    
    Args:
        image: Product image file (JPEG, PNG)
        slat_cfg_scale: SLAT configuration scale (default: 5.0)
        ss_cfg_scale: SS configuration scale (default: 10.0)
        slat_sampling_steps: SLAT sampling steps (default: 50)
        ss_sampling_steps: SS sampling steps (default: 50)
        seed: Random seed for reproducibility (default: 0)
        return_json: If True, return JSON with base64-encoded GLB; if False, return binary GLB (default: False)
        
    Returns:
        Binary GLB file (model/gltf-binary) or JSON with base64-encoded GLB
    """
    try:
        validation_result, error_response = await _validate_image(image, "/generate/3d")
        if error_response:
            return error_response
        image_bytes, content_type = validation_result
        
        logger.info(
            f"Generating 3D asset: slat_cfg={slat_cfg_scale}, ss_cfg={ss_cfg_scale}, "
            f"slat_steps={slat_sampling_steps}, ss_steps={ss_sampling_steps}, seed={seed}"
        )
        
        result = await generate_3d_asset(
            image_bytes=image_bytes,
            content_type=content_type,
            slat_cfg_scale=slat_cfg_scale,
            ss_cfg_scale=ss_cfg_scale,
            slat_sampling_steps=slat_sampling_steps,
            ss_sampling_steps=ss_sampling_steps,
            seed=seed
        )
        
        glb_data = result["glb_data"]
        artifact_id = result["artifact_id"]
        metadata = result["metadata"]
        
        logger.info(
            f"/generate/3d success: artifact_id={artifact_id} size={metadata['size_bytes']} bytes"
        )
        
        if return_json:
            # Return JSON with base64-encoded GLB
            logger.info(f"Encoding GLB to base64: {len(glb_data)} bytes")
            glb_b64 = base64.b64encode(glb_data).decode("ascii")
            b64_size = len(glb_b64)
            logger.info(f"Base64 encoded: {b64_size} chars (~{b64_size / 1024 / 1024:.2f} MB)")
            
            payload = {
                "glb_base64": glb_b64,
                "artifact_id": artifact_id,
                "metadata": metadata
            }
            
            import json as json_module
            payload_json = json_module.dumps(payload)
            payload_size = len(payload_json)
            logger.info(f"Returning JSON response with glb_base64 field (present: {bool(glb_b64)}, approx payload size: {payload_size / 1024 / 1024:.2f} MB)")            
            
            return JSONResponse(
                payload,
                headers={
                    "X-GLB-Size-Bytes": str(metadata['size_bytes']),
                    "X-Artifact-ID": artifact_id
                }
            )
        else:
            # Return binary GLB file
            return Response(
                content=glb_data,
                media_type="model/gltf-binary",
                headers={
                    "Content-Disposition": f'attachment; filename="product_3d_{artifact_id}.glb"'
                }
            )
        
    except httpx.ConnectError as exc:
        logger.exception(f"/generate/3d connection error: {exc}")
        return JSONResponse({
            "detail": "Unable to connect to the TRELLIS 3D generation endpoint. Please verify that the service is running and configured correctly."
        }, status_code=503)
    except httpx.TimeoutException as exc:
        logger.exception(f"/generate/3d timeout error: {exc}")
        return JSONResponse({
            "detail": "3D generation request timed out. The model may be overloaded or the image may be too complex."
        }, status_code=504)
    except httpx.HTTPStatusError as exc:
        logger.exception(f"/generate/3d HTTP error: {exc}")
        return JSONResponse({
            "detail": f"3D generation service returned an error: {exc.response.status_code}"
        }, status_code=exc.response.status_code)
    except Exception as exc:
        logger.exception(f"/generate/3d exception: {exc}")
        return JSONResponse({"detail": str(exc)}, status_code=500)