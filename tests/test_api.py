# API smoke tests.
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_route() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "vector_store_ready" in body
