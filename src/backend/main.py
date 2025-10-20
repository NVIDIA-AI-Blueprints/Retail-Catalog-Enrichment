import base64
import json
import logging
from contextlib import asynccontextmanager
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from langgraph.graph import StateGraph

from backend.vlm import run_vlm_analysis, vlm_node, VLMState
from backend.image import generate_image_variation, planner_node, flux_node, persist_node
from backend.trellis import generate_3d_asset

load_dotenv()

logger = logging.getLogger("catalog_enrichment.api")
compiled_graph = None
VALID_LOCALES = {"en-US", "en-GB", "en-AU", "en-CA", "es-ES", "es-MX", "es-AR", "es-CO", "fr-FR", "fr-CA"}


def create_compiled_graph():
    """Create the complete pipeline graph (VLM -> Planner -> FLUX -> Persist)."""
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

@asynccontextmanager
async def lifespan(app: FastAPI):
    global compiled_graph
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    logger.info("App startup: compiling graph (FLUX-only)")
    compiled_graph = create_compiled_graph()
    logger.info("App startup complete")
    yield

app = FastAPI(lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend origins
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
        
    except Exception as exc:
        logger.exception(f"/vlm/analyze exception: {exc}")
        return JSONResponse({"detail": str(exc)}, status_code=500)


@app.post("/generate/variation")
async def generate_variation(
    image: UploadFile = File(...),
    locale: str = Form("en-US"),
    title: str = Form(...),
    description: str = Form(...),
    categories: str = Form(...),  # JSON array as string
    tags: str = Form("[]"),  # JSON array as string
    colors: str = Form("[]"),  # JSON array as string
    enhanced_product: str = Form(None)  # Optional JSON object as string
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
            "locale": locale
        }
        
        logger.info(f"/generate/variation success: artifact_id={result['artifact_id']} image_b64_len={len(result['generated_image_b64'])}")
        return JSONResponse(payload)
        
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
        
        result = generate_3d_asset(
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
        
    except Exception as exc:
        logger.exception(f"/generate/3d exception: {exc}")
        return JSONResponse({"detail": str(exc)}, status_code=500)


@app.post("/vlm/describe")
async def vlm_describe(
    image: UploadFile = File(...), 
    locale: str = Form("en-US"),
    product_data: str = Form(None),
    brand_instructions: str = Form(None)
) -> JSONResponse:
    try:
        if locale not in VALID_LOCALES:
            logger.error(f"/vlm/describe error: invalid locale={locale}, valid options: {VALID_LOCALES}")
            return JSONResponse({"detail": f"Invalid locale. Supported locales: {sorted(VALID_LOCALES)}"}, status_code=400)

        product_json = None
        if product_data:
            try:
                product_json = json.loads(product_data)
                logger.info(f"Parsed product_data: {product_json}")
            except Exception as e:
                logger.error(f"/vlm/describe error: invalid JSON in product_data: {e}")
                return JSONResponse({"detail": f"Invalid JSON in product_data: {e}"}, status_code=400)

        validation_result, error_response = await _validate_image(image, "/vlm/describe")
        if error_response:
            return error_response
        image_bytes, content_type = validation_result

        if not compiled_graph:
            return JSONResponse({"detail": "Graph is not initialized"}, status_code=500)

        logger.info(f"Invoking graph: VLM -> Planner -> FLUX -> Persist with locale={locale} mode={'augmentation' if product_json else 'generation'} brand_instructions={bool(brand_instructions)}")
        result = compiled_graph.invoke({
            "image_bytes": image_bytes, 
            "content_type": content_type, 
            "locale": locale,
            "product_data": product_json,
            "brand_instructions": brand_instructions
        })
        logger.info("Graph invocation complete")

        if result.get("error"):
            return JSONResponse({"detail": result["error"]}, status_code=400)

        payload = result.get("enhanced_product", {}) if product_json else {
            "title": result.get("title", ""),
            "description": result.get("description", ""),
            "categories": result.get("categories", ["uncategorized"]),
            "colors": result.get("colors", []),
            "tags": result.get("tags", [])
        }
        payload["locale"] = locale
        
        # Include generated image if available
        if result.get("generated_image_b64"):
            payload["generated_image_b64"] = result.get("generated_image_b64")
            logger.info(f"Including generated image in response: b64_len={len(result.get('generated_image_b64', ''))}")
        
        if product_json:
            logger.info(f"/vlm/describe success (augmentation): keys={list(payload.keys())} locale={locale}")
        else:
            logger.info(f"/vlm/describe success (generation): title_len={len(payload['title'])} desc_len={len(payload['description'])} cats={payload['categories']} colors={payload['colors']} tags={payload['tags']} locale={locale}")
        
        return JSONResponse(payload)

    except Exception as exc:
        logger.exception(f"/vlm/describe exception: {exc}")
        return JSONResponse({"detail": str(exc)}, status_code=500)