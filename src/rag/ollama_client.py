"""Single place for resolving the Ollama client and the active chat model —
used by every module that talks to Ollama for something other than the
vision/captioning path (vision.py deliberately keeps its own resolution
since it juggles two independent roles). Before this existed, llm.py,
query_rewrite.py, structured_qa.py, and ragas_eval.py each independently
rebuilt an `ollama.Client(host=...)` and re-derived the active model —
four copies of the same two lines that had no way to stay in sync if either
ever needed to change (a timeout, an auth header, a new fallback tier).
"""
import ollama

from . import model_prefs
from .config import settings


def get_client(host: str | None = None) -> ollama.Client:
    return ollama.Client(host=host or settings.ollama_host)


def active_chat_model(model: str | None = None) -> str:
    return model or model_prefs.load_active_model("chat", settings.ollama_model)
