# catalog-enrichment

## Development setup (uv + uvicorn)

This project uses [`uv`](https://docs.astral.sh/uv/) for Python environments and dependency management and `uvicorn` for serving the ASGI backend.

### Prerequisites
- Python 3.11+
- `uv` installed (macOS/Linux):
  - `curl -LsSf https://astral.sh/uv/install.sh | sh`

### Quick start (no explicit venv)
Run the backend directly using `uv` with an isolated environment:

```
uv run uvicorn --app-dir src backend.main:app --host 0.0.0.0 --port 8000 --reload
```

### Recommended: create a local virtual environment
```
# Create and activate a local venv managed by uv
uv venv .venv
source .venv/bin/activate

# Install project dependencies from pyproject.toml
uv pip install -e .

# Run the backend
uvicorn --app-dir src backend.main:app --host 0.0.0.0 --port 8000 --reload
```

Backend endpoints:
- `GET /` → plaintext greeting
- `GET /health` → JSON health status

### VLM: Describe an image

1) Set your NVIDIA API key in `.env`:

```
cp .env .env.local  # optional; or edit .env directly
echo "NVIDIA_API_KEY=YOUR_KEY_HERE" >> .env
```

2) Start the backend (see above), then call the endpoint:

```
curl -X POST \
  -F "image=@path/to/your_image.png;type=image/png" \
  http://localhost:8000/vlm/describe
```

Response:

```
{"description":"...model output..."}
```


