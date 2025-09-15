from typing import Any
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from backend.graph import create_compiled_graph


load_dotenv()


compiled_graph: Any = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global compiled_graph
    compiled_graph = create_compiled_graph()
    yield


app: Any = FastAPI(lifespan=lifespan)


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

        if compiled_graph is None:
            return JSONResponse({"detail": "Graph is not initialized"}, status_code=500)

        state = {
            "image_bytes": image_bytes,
            "content_type": content_type,
        }
        result = compiled_graph.invoke(state)

        if result.get("error"):
            return JSONResponse({"detail": result.get("error")}, status_code=400)

        return JSONResponse({
            "title": result.get("title", ""),
            "description": result.get("description", ""),
            "categories": result.get("categories", ["uncategorized"]),
        })

    except Exception as exc:
        return JSONResponse({"detail": str(exc)}, status_code=500)


