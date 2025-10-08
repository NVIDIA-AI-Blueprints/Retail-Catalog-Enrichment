import json
import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from backend.graph import create_compiled_graph

load_dotenv()

logger = logging.getLogger("catalog_enrichment.api")
compiled_graph = None
VALID_LOCALES = {"en-US", "en-GB", "en-AU", "en-CA", "es-ES", "es-MX", "es-AR", "es-CO", "fr-FR", "fr-CA"}

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
    allow_origins=["http://localhost:3001"],  # Frontend origins
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

@app.post("/vlm/describe")
async def vlm_describe(
    image: UploadFile = File(...), 
    locale: str = Form("en-US"),
    product_data: str = Form(None)
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

        logger.info(f"Invoking graph: VLM -> Planner -> FLUX -> Persist with locale={locale} mode={'augmentation' if product_json else 'generation'}")
        result = compiled_graph.invoke({
            "image_bytes": image_bytes, 
            "content_type": content_type, 
            "locale": locale,
            "product_data": product_json
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