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
