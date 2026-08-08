import dataclasses

from src.api import main as main_module


def test_no_auth_required_by_default(client):
    # settings.api_key defaults to "" — every existing test already relies
    # on this working with no header at all; this just makes it explicit.
    assert main_module.settings.api_key == ""
    res = client.get("/api/health")
    assert res.status_code == 200


def test_rejects_missing_key_when_configured(client, monkeypatch):
    patched = dataclasses.replace(main_module.settings, api_key="secret123")
    monkeypatch.setattr(main_module, "settings", patched)

    res = client.get("/api/health")
    assert res.status_code == 401


def test_rejects_wrong_key_when_configured(client, monkeypatch):
    patched = dataclasses.replace(main_module.settings, api_key="secret123")
    monkeypatch.setattr(main_module, "settings", patched)

    res = client.get("/api/health", headers={"X-API-Key": "wrong-key"})
    assert res.status_code == 401


def test_accepts_correct_key_when_configured(client, monkeypatch):
    patched = dataclasses.replace(main_module.settings, api_key="secret123")
    monkeypatch.setattr(main_module, "settings", patched)

    res = client.get("/api/health", headers={"X-API-Key": "secret123"})
    assert res.status_code == 200


def test_static_frontend_stays_unauthenticated(client, monkeypatch):
    # The HTML/JS/CSS shell isn't sensitive on its own, and the frontend
    # needs to load *before* it can prompt for a key — only /api/* routes
    # go through the dependency (mounts bypass FastAPI's dependency system
    # entirely, this just confirms that stays true).
    patched = dataclasses.replace(main_module.settings, api_key="secret123")
    monkeypatch.setattr(main_module, "settings", patched)

    res = client.get("/")
    assert res.status_code == 200
