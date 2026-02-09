"""Integration tests for API endpoints."""
import pytest
from fastapi.testclient import TestClient

from main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


class TestHealthEndpoint:
    def test_root_returns_ok(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["service"] == "ABTGS Backend"


class TestCreateProject:
    def test_valid_url(self, client):
        resp = client.post("/api/v1/projects", json={"url": "https://yadi.sk/d/test123"})
        assert resp.status_code == 200
        data = resp.json()
        assert "id" in data
        assert len(data["id"]) == 36  # UUID

    def test_invalid_url_host(self, client):
        resp = client.post("/api/v1/projects", json={"url": "https://evil.com/file"})
        assert resp.status_code == 400
        assert "Яндекс.Диск" in resp.json()["detail"]

    def test_missing_url(self, client):
        resp = client.post("/api/v1/projects", json={})
        assert resp.status_code == 422


class TestProjectStatus:
    def test_not_found(self, client):
        resp = client.get("/api/v1/projects/nonexistent/status")
        assert resp.status_code == 404

    def test_status_after_create(self, client):
        create_resp = client.post("/api/v1/projects", json={"url": "https://yadi.sk/d/test"})
        pid = create_resp.json()["id"]
        status_resp = client.get(f"/api/v1/projects/{pid}/status")
        assert status_resp.status_code == 200
        data = status_resp.json()
        assert data["status"] in ("queued", "downloading", "error")
        assert "status_label" in data


class TestProjectResult:
    def test_not_found(self, client):
        resp = client.get("/api/v1/projects/nonexistent")
        assert resp.status_code == 404

    def test_not_completed(self, client):
        create_resp = client.post("/api/v1/projects", json={"url": "https://yadi.sk/d/test"})
        pid = create_resp.json()["id"]
        resp = client.get(f"/api/v1/projects/{pid}")
        assert resp.status_code == 400


class TestExport:
    def test_not_found(self, client):
        resp = client.post(
            "/api/v1/projects/nonexistent/export",
            json={"mappings": [], "filename": "test.docx"},
        )
        assert resp.status_code == 404
