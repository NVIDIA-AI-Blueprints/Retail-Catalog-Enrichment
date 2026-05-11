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
import time
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, Response
import httpx
from openai import APIConnectionError

from backend.policy import evaluate_policy_compliance
from backend.policy_library import PolicyLibrary
from backend.product_manual import process_manual_pdf, generate_manual_queries, extract_manual_knowledge
from backend.vlm import extract_vlm_observation, build_enriched_vlm_result, _call_nemotron_generate_faqs, _call_nemotron_extract_schema_fields
from backend.image import generate_image_variation
from backend.trellis import generate_3d_asset
from backend.config import get_config

load_dotenv()

logger = logging.getLogger("catalog_enrichment.api")
VALID_LOCALES = {"en-US", "en-GB", "en-AU", "en-CA", "es-ES", "es-MX", "es-AR", "es-CO", "fr-FR", "fr-CA"}
policy_library = PolicyLibrary()
NIM_HEALTH_CACHE_TTL_SECONDS = 30
_nim_health_cache: dict | None = None
_nim_health_cache_expires_at = 0.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    policy_library.initialize()
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
    global _nim_health_cache, _nim_health_cache_expires_at

    now = time.monotonic()
    if _nim_health_cache and now < _nim_health_cache_expires_at:
        return JSONResponse(_nim_health_cache)

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
        _nim_health_cache = result
        _nim_health_cache_expires_at = time.monotonic() + NIM_HEALTH_CACHE_TTL_SECONDS
        
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
        vlm_observation = await asyncio.to_thread(extract_vlm_observation, image_bytes, content_type, locale)

        enrichment_task = asyncio.to_thread(
            build_enriched_vlm_result,
            vlm_observation,
            locale,
            product_json,
            brand_instructions,
        )
        retrieval_task = asyncio.to_thread(
            policy_library.retrieve_context,
            {
                "title": vlm_observation.get("title", ""),
                "description": vlm_observation.get("description", ""),
                "categories": vlm_observation.get("categories", []),
                "tags": vlm_observation.get("tags", []),
                "colors": vlm_observation.get("colors", []),
            },
        )
        result, policy_contexts = await asyncio.gather(enrichment_task, retrieval_task)
        if policy_contexts:
            logger.info("Policy retrieval returned %d candidate policy record(s); running compliance evaluation.", len(policy_contexts))
            product_snapshot = {
                "locale": locale,
                "title": vlm_observation.get("title", ""),
                "description": vlm_observation.get("description", ""),
                "categories": vlm_observation.get("categories", []),
                "tags": vlm_observation.get("tags", []),
                "colors": vlm_observation.get("colors", []),
                "generated_catalog_fields": {
                    "title": result.get("title", ""),
                    "description": result.get("description", ""),
                    "categories": result.get("categories", []),
                    "tags": result.get("tags", []),
                    "colors": result.get("colors", []),
                },
                "product_data": product_json or {},
            }
            result["policy_decision"] = await asyncio.to_thread(
                evaluate_policy_compliance,
                product_snapshot,
                policy_contexts,
                locale,
            )
            logger.info(
                "Policy evaluation complete: status=%s matches=%d warnings=%d",
                result["policy_decision"].get("status"),
                len(result["policy_decision"].get("matched_policies", [])),
                len(result["policy_decision"].get("warnings", [])),
            )
        elif policy_library.list_documents():
            logger.info("Policy retrieval returned no candidates; treating loaded policies as not relevant to this product.")
            result["policy_decision"] = {
                "status": "pass",
                "label": "Policy Check Passed",
                "summary": "No loaded policy appears applicable to this product.",
                "matched_policies": [],
                "warnings": [],
                "evidence_note": "Policy retrieval did not return any candidate matches for this product.",
            }
        
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
        if result.get("policy_decision"):
            payload["policy_decision"] = result["policy_decision"]
        
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


@app.post("/vlm/faqs")
async def vlm_faqs(
    title: str = Form(""),
    description: str = Form(""),
    categories: str = Form("[]"),
    tags: str = Form("[]"),
    colors: str = Form("[]"),
    locale: str = Form("en-US"),
    manual_knowledge: str = Form(""),
) -> JSONResponse:
    """Generate FAQs from enriched product data. Called after /vlm/analyze completes.

    When *manual_knowledge* is provided (JSON dict of topic → text), the FAQ
    prompt uses both the product data and the extracted manual content to
    produce up to 10 richer FAQs.
    """
    try:
        if locale not in VALID_LOCALES:
            logger.error(f"/vlm/faqs error: invalid locale={locale}")
            return JSONResponse({"detail": f"Invalid locale. Supported locales: {sorted(VALID_LOCALES)}"}, status_code=400)

        enriched = {
            "title": title,
            "description": description,
            "categories": json.loads(categories),
            "tags": json.loads(tags),
            "colors": json.loads(colors),
        }

        parsed_knowledge = None
        if manual_knowledge and manual_knowledge.strip():
            try:
                parsed_knowledge = json.loads(manual_knowledge)
            except json.JSONDecodeError:
                logger.warning("/vlm/faqs: invalid manual_knowledge JSON, ignoring")

        faqs = await asyncio.to_thread(
            _call_nemotron_generate_faqs, enriched, locale, parsed_knowledge
        )
        return JSONResponse({"faqs": faqs})
    except (APIConnectionError, httpx.ConnectError) as exc:
        logger.exception("/vlm/faqs connection error: %s", exc)
        return JSONResponse({
            "detail": "Unable to connect to the NIM endpoint. Please verify that the NVIDIA NIM container is running."
        }, status_code=503)
    except Exception as exc:
        logger.exception("/vlm/faqs exception: %s", exc)
        return JSONResponse({"detail": str(exc)}, status_code=500)


@app.post("/vlm/manual/extract")
async def vlm_manual_extract(
    file: UploadFile = File(...),
    title: str = Form(""),
    categories: str = Form("[]"),
    locale: str = Form("en-US"),
) -> JSONResponse:
    """Extract knowledge from a product manual PDF for FAQ enrichment.

    Stateless: processes the PDF, generates product-specific queries (using
    title + categories, NOT description), retrieves relevant chunks, and
    returns the structured knowledge.  All vectors are freed after the
    response is sent.
    """
    try:
        if locale not in VALID_LOCALES:
            return JSONResponse(
                {"detail": f"Invalid locale. Supported locales: {sorted(VALID_LOCALES)}"},
                status_code=400,
            )

        if not file.filename or not file.filename.lower().endswith(".pdf"):
            return JSONResponse({"detail": "Only PDF files are accepted."}, status_code=400)

        if file.content_type and file.content_type != "application/pdf":
            return JSONResponse({"detail": "Only PDF files are accepted."}, status_code=400)

        pdf_bytes = await file.read()
        if not pdf_bytes:
            return JSONResponse({"detail": "Uploaded file is empty."}, status_code=400)

        max_pdf_size = 50 * 1024 * 1024  # 50 MB
        if len(pdf_bytes) > max_pdf_size:
            return JSONResponse({"detail": "PDF file exceeds the 50 MB size limit."}, status_code=400)

        safe_title = title[:500] if title else ""
        try:
            parsed_categories = json.loads(categories) if categories else []
            if not isinstance(parsed_categories, list):
                parsed_categories = []
            parsed_categories = [str(c)[:100] for c in parsed_categories[:20]]
        except json.JSONDecodeError:
            parsed_categories = []

        def _extract():
            ctx = process_manual_pdf(pdf_bytes, file.filename)
            queries = generate_manual_queries(safe_title, parsed_categories, locale)
            if not queries:
                logger.warning("[Manual] LLM returned no queries; returning empty knowledge")
                return ctx.filename, ctx.chunk_count, {}
            knowledge = extract_manual_knowledge(ctx, queries)
            return ctx.filename, ctx.chunk_count, knowledge

        filename, chunk_count, knowledge = await asyncio.to_thread(_extract)

        return JSONResponse({
            "filename": filename,
            "chunk_count": chunk_count,
            "knowledge": knowledge,
        })

    except ValueError as exc:
        logger.warning("/vlm/manual/extract validation error: %s", exc)
        return JSONResponse({"detail": str(exc)}, status_code=400)
    except (APIConnectionError, httpx.ConnectError) as exc:
        logger.exception("/vlm/manual/extract connection error: %s", exc)
        return JSONResponse({
            "detail": "Unable to connect to the NIM endpoint. Please verify that the NVIDIA NIM container is running."
        }, status_code=503)
    except Exception as exc:
        logger.exception("/vlm/manual/extract exception: %s", exc)
        return JSONResponse({"detail": str(exc)}, status_code=500)


@app.get("/policies")
async def list_policies() -> JSONResponse:
    try:
        return JSONResponse({"documents": policy_library.list_documents()})
    except Exception as exc:
        logger.exception("/policies list exception: %s", exc)
        return JSONResponse({"detail": str(exc)}, status_code=500)


@app.post("/policies")
async def upload_policies(
    files: list[UploadFile] = File(...),
    locale: str = Form("en-US"),
) -> JSONResponse:
    try:
        if locale not in VALID_LOCALES:
            return JSONResponse({"detail": f"Invalid locale. Supported locales: {sorted(VALID_LOCALES)}"}, status_code=400)

        uploads, error_response = await _validate_policy_uploads(files, "/policies")
        if error_response:
            return error_response

        results = policy_library.ingest_documents(uploads, locale=locale)
        return JSONResponse({"documents": policy_library.list_documents(), "results": results})
    except Exception as exc:
        logger.exception("/policies upload exception: %s", exc)
        return JSONResponse({"detail": str(exc)}, status_code=500)


@app.delete("/policies")
async def clear_policies() -> JSONResponse:
    try:
        policy_library.clear()
        return JSONResponse({"status": "ok"})
    except Exception as exc:
        logger.exception("/policies clear exception: %s", exc)
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
            locale=locale
        )
        
        payload = {
            "generated_image_b64": result["generated_image_b64"],
            "variation_plan": result["variation_plan"],
            "quality_score": result["quality_score"],
            "quality_issues": result["quality_issues"],
            "locale": locale
        }
        
        logger.info(f"/generate/variation success: image_b64_len={len(result['generated_image_b64'])} quality_score={result['quality_score']} issues_count={len(result['quality_issues'])}")
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


async def _validate_policy_uploads(policy_files: list[UploadFile], endpoint: str):
    if not policy_files:
        return None, JSONResponse({"detail": "At least one PDF file is required"}, status_code=400)

    uploads = []
    invalid_files = []

    for policy_file in policy_files:
        logger.info(
            "POST %s policy filename=%s content_type=%s",
            endpoint,
            getattr(policy_file, "filename", None),
            getattr(policy_file, "content_type", None),
        )

        filename = getattr(policy_file, "filename", None) or "policy.pdf"
        content_type = getattr(policy_file, "content_type", None) or "application/pdf"
        if content_type != "application/pdf" and not filename.lower().endswith(".pdf"):
            invalid_files.append(filename)
            continue

        pdf_bytes = await policy_file.read()
        if not pdf_bytes:
            invalid_files.append(filename)
            continue
        uploads.append({"filename": filename, "bytes": pdf_bytes})

    if invalid_files:
        return None, JSONResponse(
            {"detail": f"Policy files must be non-empty PDFs. Invalid files: {', '.join(sorted(invalid_files))}"},
            status_code=400,
        )

    return uploads, None


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


# ---------------------------------------------------------------------------
# Protocol schema endpoints
# ---------------------------------------------------------------------------

def _empty_money():
    return {"amount": None, "currency": None}


def _empty_measurement():
    return {"value": None, "unit": None}


def _build_acp_schema(enriched: dict, faqs: list, extracted: dict) -> dict:
    """Build an ACP schema instance, merging enriched data with LLM-extracted fields."""
    details = extracted.get("product_details", [])
    highlights = extracted.get("product_highlights", [])

    return {
        "product": {
            "id": None,
            "title": enriched.get("title", ""),
            "description": enriched.get("description", ""),
            "brand": extracted.get("brand"),
            "attributes": {
                "colors": enriched.get("colors", []),
                "material": extracted.get("material"),
                "size": None,
                "weight": _empty_measurement(),
                "condition": extracted.get("condition", "new"),
                "age_group": extracted.get("age_group"),
                "gender": extracted.get("gender"),
                "pattern": None,
            },
            "categories": enriched.get("categories", []),
            "tags": enriched.get("tags", []),
            "images": {
                "primary": None,
                "additional": [],
                "lifestyle": None,
                "video": None,
                "virtual_model_3d": None,
            },
            "identifiers": {
                "gtin": None,
                "mpn": None,
                "sku": None,
            },
            "dimensions": {
                "length": _empty_measurement(),
                "width": _empty_measurement(),
                "height": _empty_measurement(),
                "weight": _empty_measurement(),
            },
            "details": details if isinstance(details, list) else [],
            "highlights": highlights if isinstance(highlights, list) else [],
        },
        "pricing": {
            "currency": None,
            "price": None,
            "sale_price": None,
            "sale_price_effective_date": None,
            "cost_of_goods_sold": None,
            "availability": "in_stock",
            "availability_date": None,
            "expiration_date": None,
            "installment": {
                "months": None,
                "amount": _empty_money(),
                "downpayment": _empty_money(),
                "credit_type": None,
            },
            "subscription_cost": {
                "period": None,
                "period_length": None,
                "amount": _empty_money(),
            },
            "unit_pricing": {
                "measure": _empty_measurement(),
                "base_measure": _empty_measurement(),
            },
            "loyalty_program": [],
            "auto_pricing_min_price": None,
            "maximum_retail_price": None,
        },
        "faqs": [{"question": f["question"], "answer": f["answer"]} for f in faqs],
        "agent_actions": {
            "discoverable": True,
            "buyable": True,
            "returnable": True,
            "comparable": True,
            "subscribable": False,
        },
        "fulfillment": {
            "shipping": {
                "eligible": None,
                "weight": _empty_measurement(),
                "length": _empty_measurement(),
                "width": _empty_measurement(),
                "height": _empty_measurement(),
                "label": None,
                "ships_from_country": None,
                "min_handling_time": None,
                "max_handling_time": None,
                "transit_business_days": [],
                "handling_business_days": [],
                "free_shipping_threshold": [],
                "rules": [],
                "carrier_rules": [],
            },
            "pickup_eligible": None,
            "digital_delivery": None,
            "handling_cutoff_time": [],
            "minimum_order_value": [],
            "return_policy_label": None,
        },
        "campaigns": {
            "ads_redirect": None,
            "custom_labels": [None, None, None, None, None],
            "promotion_ids": [],
            "short_title": extracted.get("short_title"),
            "excluded_destinations": [],
            "included_destinations": [],
            "shopping_ads_excluded_countries": [],
            "pause": None,
        },
        "certifications": [],
        "energy_efficiency": {
            "class": None,
            "min_class": None,
            "max_class": None,
        },
        "bundling": {
            "multipack": None,
            "is_bundle": False,
        },
        "marketplace": {
            "external_seller_id": None,
        },
        "metadata": {
            "locale": None,
            "enrichment_source": "nvidia-catalog-enrichment",
            "generated_at": None,
        },
    }


def _build_ucp_schema(enriched: dict, faqs: list, extracted: dict) -> dict:
    """Build a UCP schema instance, merging enriched data with LLM-extracted fields."""
    colors = enriched.get("colors", [])
    categories = enriched.get("categories", [])
    tags = enriched.get("tags", [])
    details = extracted.get("product_details", [])
    highlights = extracted.get("product_highlights", tags)

    return {
        # ── Basic product data ──
        "id": None,
        "title": None,
        "structured_title": {
            "digital_source_type": "trained_algorithmic_media",
            "content": enriched.get("title", ""),
        },
        "description": None,
        "structured_description": {
            "digital_source_type": "trained_algorithmic_media",
            "content": enriched.get("description", ""),
        },
        "link": None,
        "image_link": None,
        "additional_image_link": [],
        "video_link": None,
        "virtual_model_link": None,
        "mobile_link": None,

        # ── Price and availability ──
        "availability": "in_stock",
        "availability_date": None,
        "cost_of_goods_sold": _empty_money(),
        "expiration_date": None,
        "price": _empty_money(),
        "sale_price": _empty_money(),
        "sale_price_effective_date": None,
        "unit_pricing_measure": _empty_measurement(),
        "unit_pricing_base_measure": _empty_measurement(),
        "installment": {
            "months": None,
            "amount": _empty_money(),
            "downpayment": _empty_money(),
            "credit_type": None,
        },
        "subscription_cost": {
            "period": None,
            "period_length": None,
            "amount": _empty_money(),
        },
        "loyalty_program": [],
        "auto_pricing_min_price": _empty_money(),
        "maximum_retail_price": _empty_money(),

        # ── Product category ──
        "google_product_category": extracted.get("google_product_category"),
        "product_type": " > ".join(categories) if categories else None,

        # ── Product identifiers ──
        "brand": extracted.get("brand"),
        "gtin": None,
        "mpn": None,
        "identifier_exists": False,

        # ── Detailed product description ──
        "condition": extracted.get("condition", "new"),
        "adult": False,
        "multipack": None,
        "is_bundle": False,
        "certification": [],
        "energy_efficiency_class": None,
        "min_energy_efficiency_class": None,
        "max_energy_efficiency": None,
        "age_group": extracted.get("age_group"),
        "color": " / ".join(colors) if colors else None,
        "gender": extracted.get("gender"),
        "material": extracted.get("material"),
        "pattern": None,
        "size": None,
        "size_type": [],
        "size_system": None,
        "item_group_id": None,
        "product_length": _empty_measurement(),
        "product_width": _empty_measurement(),
        "product_height": _empty_measurement(),
        "product_weight": _empty_measurement(),
        "product_detail": details if isinstance(details, list) else [],
        "product_highlight": highlights if isinstance(highlights, list) else [],
        "faqs": [{"question": f["question"], "answer": f["answer"]} for f in faqs],

        # ── Shopping campaigns and other configurations ──
        "ads_redirect": None,
        "custom_label_0": None,
        "custom_label_1": None,
        "custom_label_2": None,
        "custom_label_3": None,
        "custom_label_4": None,
        "promotion_id": [],
        "lifestyle_image_link": None,
        "short_title": extracted.get("short_title"),

        # ── Marketplaces ──
        "external_seller_id": None,

        # ── Destinations ──
        "excluded_destination": [],
        "included_destination": [],
        "shopping_ads_excluded_country": [],
        "pause": None,

        # ── Shipping and returns ──
        "shipping": [],
        "carrier_shipping": [],
        "handling_cutoff_time": [],
        "minimum_order_value": [],
        "shipping_label": None,
        "shipping_weight": _empty_measurement(),
        "shipping_length": _empty_measurement(),
        "shipping_width": _empty_measurement(),
        "shipping_height": _empty_measurement(),
        "ships_from_country": None,
        "max_handling_time": None,
        "min_handling_time": None,
        "shipping_transit_business_days": [],
        "shipping_handling_business_days": [],
        "free_shipping_threshold": [],
        "return_policy_label": None,
    }


@app.post("/protocols/generate")
async def protocols_generate(
    title: str = Form(""),
    description: str = Form(""),
    categories: str = Form("[]"),
    tags: str = Form("[]"),
    colors: str = Form("[]"),
    faqs: str = Form("[]"),
    locale: str = Form("en-US"),
) -> JSONResponse:
    """Generate both ACP and UCP schemas from enriched product data.

    Calls the LLM once to extract structured fields (brand, material,
    product_details, etc.), then builds both schemas from the same
    extraction. Returns ``{"acp": {...}, "ucp": {...}}``.
    """
    try:
        enriched = {
            "title": title,
            "description": description,
            "categories": json.loads(categories),
            "tags": json.loads(tags),
            "colors": json.loads(colors),
        }
        parsed_faqs = json.loads(faqs)

        extracted = await asyncio.to_thread(
            _call_nemotron_extract_schema_fields, enriched, locale
        )

        acp = _build_acp_schema(enriched, parsed_faqs, extracted)
        ucp = _build_ucp_schema(enriched, parsed_faqs, extracted)

        return JSONResponse({"acp": acp, "ucp": ucp})
    except (APIConnectionError, httpx.ConnectError) as exc:
        logger.exception("/protocols/generate connection error: %s", exc)
        return JSONResponse({
            "detail": "Unable to connect to the NIM endpoint. Please verify that the NVIDIA NIM container is running."
        }, status_code=503)
    except json.JSONDecodeError as exc:
        logger.error("/protocols/generate JSON parse error: %s", exc)
        return JSONResponse({"detail": "Invalid JSON in request fields."}, status_code=400)
    except Exception as exc:
        logger.exception("/protocols/generate exception: %s", exc)
        return JSONResponse({"detail": str(exc)}, status_code=500)
