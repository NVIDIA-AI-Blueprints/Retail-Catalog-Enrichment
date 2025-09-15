from fastapi.testclient import TestClient

from backend.main import app


client = TestClient(app)


def test_homepage():
    response = client.get("/")
    assert response.status_code == 200
    assert response.text == "Catalog Enrichment Backend"


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


