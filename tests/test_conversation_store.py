from src.rag import conversation_store


def test_load_history_returns_empty_for_unknown_conversation():
    assert conversation_store.load_history("nonexistent-conversation-xyz") == []


def test_save_and_load_roundtrip():
    conv_id = "test-conv-roundtrip"
    history = [
        {"role": "user", "content": "How tall is the Eiffel Tower?"},
        {"role": "assistant", "content": "330 metres."},
    ]
    conversation_store.save_history(conv_id, history)
    assert conversation_store.load_history(conv_id) == history


def test_save_overwrites_previous_history():
    conv_id = "test-conv-overwrite"
    conversation_store.save_history(conv_id, [{"role": "user", "content": "first"}])
    conversation_store.save_history(conv_id, [{"role": "user", "content": "second"}])
    assert conversation_store.load_history(conv_id) == [{"role": "user", "content": "second"}]


def test_clear_history_removes_conversation():
    conv_id = "test-conv-clear"
    conversation_store.save_history(conv_id, [{"role": "user", "content": "hi"}])
    conversation_store.clear_history(conv_id)
    assert conversation_store.load_history(conv_id) == []


def test_clear_history_on_unknown_conversation_does_not_raise():
    conversation_store.clear_history("never-existed-conversation")


def test_prune_removes_only_old_conversations(monkeypatch):
    import sqlite3
    from datetime import datetime, timedelta, timezone

    fresh_id, stale_id = "test-conv-fresh", "test-conv-stale"
    conversation_store.save_history(fresh_id, [{"role": "user", "content": "recent"}])
    conversation_store.save_history(stale_id, [{"role": "user", "content": "old"}])

    # Backdate the "stale" row's updated_at directly — save_history() always
    # stamps "now", so this is the only way to simulate an old conversation.
    old_timestamp = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
    conn = sqlite3.connect(conversation_store.DB_PATH)
    conn.execute("UPDATE conversations SET updated_at = ? WHERE conversation_id = ?", (old_timestamp, stale_id))
    conn.commit()
    conn.close()

    removed = conversation_store.prune_old_conversations(max_age_days=30)

    assert removed >= 1
    assert conversation_store.load_history(fresh_id) != []
    assert conversation_store.load_history(stale_id) == []
