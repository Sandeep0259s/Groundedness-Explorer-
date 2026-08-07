import json

from src.rag import model_prefs


def test_role_based_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(model_prefs, "MODEL_PREF_FILE", tmp_path / ".model_choice.json")

    # A role that's never been set falls back to the caller-supplied default.
    assert model_prefs.load_active_model("chat", "default-model") == "default-model"

    model_prefs.save_active_model("chat", "llama3.2:3b")
    model_prefs.save_active_model("vision", "moondream")

    assert model_prefs.load_active_model("chat") == "llama3.2:3b"
    assert model_prefs.load_active_model("vision") == "moondream"
    # Adding a new role never touches the ones already stored.
    assert model_prefs.load_active_model("reranker", "some-default") == "some-default"

    data = json.loads((tmp_path / ".model_choice.json").read_text())
    assert data == {"chat": "llama3.2:3b", "vision": "moondream"}


def test_load_survives_missing_or_corrupt_file(tmp_path, monkeypatch):
    pref_file = tmp_path / ".model_choice.json"
    monkeypatch.setattr(model_prefs, "MODEL_PREF_FILE", pref_file)
    assert model_prefs.load_active_model("chat", "fallback") == "fallback"

    pref_file.write_text("not valid json")
    assert model_prefs.load_active_model("chat", "fallback") == "fallback"
