import io
from unittest.mock import patch
from fastapi.testclient import TestClient
from backend.main import app


client = TestClient(app)


def test_vlm_describe_happy_path_png():
    fake_image = io.BytesIO(b"\x89PNG\r\n\x1a\nFAKEPNGDATA")

    # Mock compiled_graph.invoke to return a deterministic result
    with patch("backend.main.compiled_graph") as mock_graph:
        mock_graph.invoke.return_value = {
            "title": "Modern Office Chair",
            "description": "Ergonomic mesh back chair with adjustable height.",
            "categories": ["clothing", "electronics"],
        }

        response = client.post(
            "/vlm/describe",
            files={
                "image": ("test.png", fake_image, "image/png"),
            },
        )

    assert response.status_code == 200
    data = response.json()
    assert set(data.keys()) == {"title", "description", "categories"}
    assert data["title"] == "Modern Office Chair"
    assert data["description"].startswith("Ergonomic")
    assert isinstance(data["categories"], list)


def test_vlm_describe_rejects_non_image_content_type():
    fake_file = io.BytesIO(b"not an image")
    response = client.post(
        "/vlm/describe",
        files={
            "image": ("test.txt", fake_file, "text/plain"),
        },
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "File must be an image"


def test_vlm_describe_empty_file():
    empty_image = io.BytesIO(b"")
    response = client.post(
        "/vlm/describe",
        files={
            "image": ("empty.png", empty_image, "image/png"),
        },
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Uploaded file is empty"


def test_vlm_describe_uninitialized_graph():
    fake_image = io.BytesIO(b"\x89PNG\r\n\x1a\nFAKEPNGDATA")

    # Ensure compiled_graph is None during this call
    with patch("backend.main.compiled_graph", None):
        response = client.post(
            "/vlm/describe",
            files={
                "image": ("test.png", fake_image, "image/png"),
            },
        )

    assert response.status_code == 500
    assert response.json()["detail"] == "Graph is not initialized"


