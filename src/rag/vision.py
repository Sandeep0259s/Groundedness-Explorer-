"""Image captioning (at ingest time) and direct visual question-answering
(at ask time) via whichever vision-capable Ollama model is currently active.

No model name is hardcoded anywhere here: `active_vision_model()` resolves
it fresh on every call — an explicit RAG_VISION_MODEL env var wins, then
whatever the UI/API last set via `model_prefs`, then the first vision-
capable model model_registry finds already pulled. Pulling a different or
newer vision model and selecting it in the Model panel takes effect
immediately, with no restart and no code change.
"""
import base64
from functools import lru_cache
from pathlib import Path

import ollama

from . import model_prefs, model_registry
from .config import settings

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}

CAPTION_PROMPT = (
    "Describe this image factually and in detail: any visible text, objects, "
    "people, colors, setting, and composition. Do not speculate beyond what's visible."
)


class VisionUnavailable(RuntimeError):
    """Raised when no vision-capable model is pulled/selected — callers
    should degrade gracefully (skip captioning, fall back to text-only
    answering) rather than treat this as a hard failure."""


def active_vision_model() -> str | None:
    if settings.vision_model:
        return settings.vision_model
    preferred = model_prefs.load_active_model("vision")
    if preferred:
        return preferred
    return model_registry.first_pulled(model_registry.models_with_capability("vision"))


def _encode(image_path: Path) -> str:
    return base64.b64encode(image_path.read_bytes()).decode()


class VisionModel:
    def __init__(self, host: str = settings.ollama_host):
        self.client = ollama.Client(host=host)

    def caption(self, image_path: Path) -> str:
        model = active_vision_model()
        if not model:
            raise VisionUnavailable("no vision-capable Ollama model is pulled")
        response = self.client.chat(
            model=model,
            messages=[{"role": "user", "content": CAPTION_PROMPT, "images": [_encode(image_path)]}],
            think=False,  # extended reasoning adds real latency on CPU with no benefit for captioning
        )
        return response["message"]["content"]

    def answer(self, image_path: Path, question: str, context: list[str] | None = None) -> str:
        model = active_vision_model()
        if not model:
            raise VisionUnavailable("no vision-capable Ollama model is pulled")
        extra = f"\n\nAdditional retrieved context:\n{chr(10).join(context)}" if context else ""
        prompt = (
            f"Look at the image and answer the question using what you actually see in it.{extra}"
            f"\n\nQuestion: {question}"
        )
        response = self.client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt, "images": [_encode(image_path)]}],
            think=False,
        )
        return response["message"]["content"]


@lru_cache(maxsize=1)
def get_vision_model() -> VisionModel:
    return VisionModel()
