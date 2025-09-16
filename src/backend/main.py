from contextlib import asynccontextmanager
import logging

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, PlainTextResponse
from backend.graph import create_compiled_graph

load_dotenv()


logger = logging.getLogger("catalog_enrichment.api")

compiled_graph = None


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
    
    content_type = (image.content_type or "image/png") if hasattr(image, "content_type") else "image/png"
    if not content_type.startswith("image/"):
        logger.error(f"{endpoint} error: non-image content_type=%s", content_type)
        return None, JSONResponse({"detail": "File must be an image"}, status_code=400)
    
    return (image_bytes, content_type), None


@app.post("/vlm/describe")
async def vlm_describe(image: UploadFile = File(...)) -> JSONResponse:
    try:
        validation_result, error_response = await _validate_image(image, "/vlm/describe")
        if error_response:
            return error_response
        image_bytes, content_type = validation_result

        if compiled_graph is None:
            return JSONResponse({"detail": "Graph is not initialized"}, status_code=500)

        logger.info("Invoking graph: VLM -> Planner -> FLUX -> Persist")
        result = compiled_graph.invoke({"image_bytes": image_bytes, "content_type": content_type})
        logger.info("Graph invocation complete")

        if result.get("error"):
            return JSONResponse({"detail": result.get("error")}, status_code=400)

        payload = {
            "title": result.get("title", ""),
            "description": result.get("description", ""),
            "categories": result.get("categories", ["uncategorized"])
        }
        logger.info("/vlm/describe success: title_len=%d desc_len=%d cats=%s",
                   len(payload["title"]), len(payload["description"]), payload["categories"])
        return JSONResponse(payload)

    except Exception as exc:
        logger.exception("/vlm/describe exception: %s", exc)
        return JSONResponse({"detail": str(exc)}, status_code=500)

@app.post("/image/variation")
async def image_variation(image: UploadFile = File(...)) -> JSONResponse:
    try:
        validation_result, error_response = await _validate_image(image, "/image/variation")
        if error_response:
            return error_response
        image_bytes, content_type = validation_result

        if compiled_graph is None:
            return JSONResponse({"detail": "Graph is not initialized"}, status_code=500)

        logger.info("Invoking graph: VLM -> Planner -> FLUX -> Persist")
        result = compiled_graph.invoke({"image_bytes": image_bytes, "content_type": content_type})
        logger.info("Graph invocation complete")

        if result.get("error"):
            return JSONResponse({"detail": result.get("error")}, status_code=400)

        payload = {
            "title": result.get("title", ""),
            "description": result.get("description", ""),
            "categories": result.get("categories", ["uncategorized"]),
            "artifact_id": result.get("artifact_id"),
            "image_path": result.get("image_path"),
            "metadata_path": result.get("metadata_path"),
            "variation_plan": result.get("variation_plan")
        }
        logger.info("/image/variation success: artifact_id=%s", payload.get("artifact_id"))
        return JSONResponse(payload)

    except Exception as exc:
        logger.exception("/image/variation exception: %s", exc)
        return JSONResponse({"detail": str(exc)}, status_code=500)

