import json
from typing import Any, Dict


async def app(scope: Dict[str, Any], receive, send) -> None:
    """Minimal ASGI application served by uvicorn.

    Endpoints:
    - GET /          -> 200 text/plain greeting
    - GET /health    -> 200 application/json {"status":"ok"}
    """

    if scope.get("type") != "http":
        # Only handle HTTP requests for now
        return

    path = scope.get("path", "/")
    method = scope.get("method", "GET").upper()

    if method == "GET" and path == "/health":
        body_bytes = json.dumps({"status": "ok"}).encode("utf-8")
        headers = [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body_bytes)).encode("ascii")),
        ]
        await send({"type": "http.response.start", "status": 200, "headers": headers})
        await send({"type": "http.response.body", "body": body_bytes})
        return

    if method == "GET" and path == "/":
        body_bytes = b"Catalog Enrichment Backend"
        headers = [
            (b"content-type", b"text/plain; charset=utf-8"),
            (b"content-length", str(len(body_bytes)).encode("ascii")),
        ]
        await send({"type": "http.response.start", "status": 200, "headers": headers})
        await send({"type": "http.response.body", "body": body_bytes})
        return

    # Fallback 404
    body_bytes = json.dumps({"detail": "Not Found"}).encode("utf-8")
    headers = [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(body_bytes)).encode("ascii")),
    ]
    await send({"type": "http.response.start", "status": 404, "headers": headers})
    await send({"type": "http.response.body", "body": body_bytes})


