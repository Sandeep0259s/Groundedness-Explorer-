import json
import time

import pytest
from fastapi.testclient import TestClient

from src.rag.config import settings


def _wait_for_job(client: TestClient, job_id: str, timeout: float = 30.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("done", "error"):
            return job
        time.sleep(0.5)
    raise TimeoutError(f"job {job_id} did not finish within {timeout}s")


def test_health(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


def test_default_labels_present(client):
    res = client.get("/api/labels")
    names = {label["name"] for label in res.json()["labels"]}
    assert "general" in names
    assert "session" in names


def test_create_list_delete_label(client):
    res = client.post("/api/labels", json={"name": "coursework"})
    assert res.status_code == 200
    assert res.json()["name"] == "coursework"

    names = {label["name"] for label in client.get("/api/labels").json()["labels"]}
    assert "coursework" in names

    res = client.delete("/api/labels/coursework")
    assert res.status_code == 200
    names = {label["name"] for label in res.json()["labels"]}
    assert "coursework" not in names


def test_cannot_delete_general_label(client):
    res = client.delete("/api/labels/general")
    assert res.status_code == 400


def test_invalid_label_name_rejected(client):
    res = client.post("/api/labels", json={"name": "has spaces"})
    assert res.status_code == 400


def test_upload_ingest_and_list_document(client):
    content = b"The Great Wall of China is located in northern China."
    res = client.post(
        "/api/upload",
        files={"files": ("wall.txt", content, "text/plain")},
        data={"label": "geography"},
    )
    assert res.status_code == 200
    job_id = res.json()["job_id"]

    job = _wait_for_job(client, job_id)
    assert job["status"] == "done"
    assert job["results"][0]["chunks"] > 0

    docs = client.get("/api/documents", params={"label": "geography"}).json()
    sources = [s["source"] for s in docs["sources"]]
    assert any("wall.txt" in s for s in sources)

    client.delete("/api/labels/geography")


def test_upload_rejects_unsupported_extension(client):
    res = client.post("/api/upload", files={"files": ("notes.exe", b"binary", "application/octet-stream")})
    assert res.status_code == 200
    assert res.json()["skipped"][0]["error"] == "unsupported file type"


def test_delete_single_document(client):
    content = b"Temporary document for a delete test."
    res = client.post("/api/upload", files={"files": ("temp.txt", content, "text/plain")})
    job = _wait_for_job(client, res.json()["job_id"])
    source = job["sources"][0]["source"] if job["sources"] else None
    assert source is not None

    res = client.request("DELETE", "/api/documents", params={"source": source})
    assert res.status_code == 200
    sources = [s["source"] for s in res.json()["sources"]]
    assert source not in sources


def test_ask_without_ingested_documents_is_graceful(client):
    # A label with nothing in it should get a clear "no context" answer, not an error.
    res = client.post("/api/ask", json={"question": "anything?", "label": "empty-label-xyz"})
    assert res.status_code == 200
    body = res.json()
    assert body["groundedness"]["label"] == "unknown"
    assert body["answer_mode"] == "text"


def test_documents_file_rejects_path_outside_data_dir(client):
    res = client.get("/api/documents/file", params={"source": "../../../windows/win.ini"})
    assert res.status_code == 400


def test_documents_file_rejects_non_image(client):
    res = client.get("/api/documents/file", params={"source": f"{settings.data_dir}/eiffel_tower.txt"})
    assert res.status_code in (400, 415, 404)


def test_ask_empty_question_rejected(client):
    res = client.post("/api/ask", json={"question": "   "})
    assert res.status_code == 400


def test_ask_and_multiturn_conversation(client, ollama_available):
    if not ollama_available:
        pytest.skip("Ollama is not running — skipping tests that need real LLM generation")

    content = b"Mount Kilimanjaro is the tallest mountain in Africa, standing at 5895 metres."
    upload = client.post("/api/upload", files={"files": ("mountain.txt", content, "text/plain")}, data={"label": "geo2"})
    _wait_for_job(client, upload.json()["job_id"])

    res1 = client.post("/api/ask", json={"question": "How tall is Mount Kilimanjaro?", "label": "geo2"})
    assert res1.status_code == 200
    body1 = res1.json()
    assert "conversation_id" in body1

    res2 = client.post("/api/ask", json={
        "question": "Which continent is it on?",
        "label": "geo2",
        "conversation_id": body1["conversation_id"],
    })
    assert res2.status_code == 200


def test_ask_stream_emits_tokens_then_done(client, ollama_available):
    if not ollama_available:
        pytest.skip("Ollama is not running — skipping tests that need real LLM generation")

    content = b"The Great Barrier Reef is located off the coast of Queensland, Australia."
    upload = client.post("/api/upload", files={"files": ("reef.txt", content, "text/plain")}, data={"label": "geo3"})
    _wait_for_job(client, upload.json()["job_id"])

    with client.stream(
        "POST", "/api/ask/stream", json={"question": "Where is the Great Barrier Reef?", "label": "geo3"}
    ) as res:
        assert res.status_code == 200
        raw = "".join(res.iter_text())

    assert "event: token" in raw
    assert "event: done" in raw
    done_payload = raw.split("event: done\ndata: ")[1].strip()
    body = json.loads(done_payload)
    assert body["answer"]
    assert "groundedness" in body
    assert "answer_mode" in body

    client.delete("/api/labels/geo3")

    client.delete("/api/labels/geo2")


def test_system_set_model_rejects_bad_role(client, monkeypatch, tmp_path):
    from src.rag import model_prefs

    monkeypatch.setattr(model_prefs, "MODEL_PREF_FILE", tmp_path / ".model_choice.json")
    res = client.post("/api/system/model", json={"model": "llama3.2:3b", "role": "bogus"})
    assert res.status_code == 400


def test_system_set_model_defaults_role_to_chat(client, monkeypatch, tmp_path):
    from src.rag import model_prefs

    monkeypatch.setattr(model_prefs, "MODEL_PREF_FILE", tmp_path / ".model_choice.json")
    res = client.post("/api/system/model", json={"model": "llama3.2:3b"})
    assert res.status_code == 200
    assert res.json() == {"role": "chat", "active": "llama3.2:3b"}


def test_system_models_endpoint_reports_capabilities_and_active(client, ollama_available, monkeypatch, tmp_path):
    if not ollama_available:
        pytest.skip("Ollama is not running — skipping tests that need a real Ollama server")

    from src.rag import model_prefs

    monkeypatch.setattr(model_prefs, "MODEL_PREF_FILE", tmp_path / ".model_choice.json")
    res = client.get("/api/system/models")
    assert res.status_code == 200
    body = res.json()
    assert isinstance(body["models"], list)
    if body["models"]:
        assert "capabilities" in body["models"][0]
    assert "chat" in body["active"]
    assert "vision_caption" in body["active"]
    assert "vision_answer" in body["active"]
