"""Image captioning (at ingest time) and direct visual question-answering
(at ask time) via whichever vision-capable Ollama model is currently active.

No model name is hardcoded anywhere here: `active_vision_model()` resolves
it fresh on every call — an explicit RAG_VISION_MODEL env var wins, then
whatever the UI/API last set via `model_prefs`, then the first vision-
capable model model_registry finds already pulled. Pulling a different or
newer vision model and selecting it in the Model panel takes effect
immediately, with no restart and no code change.

Captioning and answering resolve their active model *independently* (roles
"caption" and "answer") — testing found a small model can be excellent at
open-ended captioning while being unreliable on terse direct questions, so
forcing one model to do both jobs well is the wrong assumption to bake in.
"""
import base64
from collections.abc import Iterator
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


def active_vision_model(role: str = "answer") -> str | None:
    """role: "caption" (ingest-time) or "answer" (ask-time) — each resolves
    to its own preference so they can point at different models."""
    if settings.vision_model:
        return settings.vision_model
    preferred = model_prefs.load_active_model(f"vision_{role}")
    if preferred:
        return preferred
    return model_registry.first_pulled(model_registry.models_with_capability("vision"))


def _encode_bytes(data: bytes) -> str:
    return base64.b64encode(data).decode()


def _encode(image_path: Path) -> str:
    return _encode_bytes(image_path.read_bytes())


class VisionModel:
    def __init__(self, host: str = settings.ollama_host):
        self.client = ollama.Client(host=host)

    def caption(self, image_path: Path) -> str:
        return self.caption_bytes(image_path.read_bytes())

    def caption_bytes(self, image_bytes: bytes) -> str:
        """Same as caption(), but for image data that isn't (and doesn't
        need to be) written to disk first — e.g. a video keyframe pulled
        straight out of the decoded stream."""
        model = active_vision_model("caption")
        if not model:
            raise VisionUnavailable("no vision-capable Ollama model is pulled")
        response = self.client.chat(
            model=model,
            messages=[{"role": "user", "content": CAPTION_PROMPT, "images": [_encode_bytes(image_bytes)]}],
            think=False,  # extended reasoning adds real latency on CPU with no benefit for captioning
        )
        return response["message"]["content"]

    def _answer_prompt(self, question: str, context: list[str] | None) -> str:
        extra = f"\n\nAdditional retrieved context:\n{chr(10).join(context)}" if context else ""
        return (
            f"Look at the image and answer the question using what you actually see in it.{extra}"
            f"\n\nQuestion: {question}"
        )

    def answer(self, image_path: Path, question: str, context: list[str] | None = None) -> str:
        model = active_vision_model("answer")
        if not model:
            raise VisionUnavailable("no vision-capable Ollama model is pulled")
        response = self.client.chat(
            model=model,
            messages=[{"role": "user", "content": self._answer_prompt(question, context), "images": [_encode(image_path)]}],
            think=False,
        )
        return response["message"]["content"]

    def answer_stream(self, image_path: Path, question: str, context: list[str] | None = None) -> Iterator[str]:
        model = active_vision_model("answer")
        if not model:
            raise VisionUnavailable("no vision-capable Ollama model is pulled")
        stream = self.client.chat(
            model=model,
            messages=[{"role": "user", "content": self._answer_prompt(question, context), "images": [_encode(image_path)]}],
            think=False,
            stream=True,
        )
        for chunk in stream:
            if chunk.message.content:
                yield chunk.message.content


@lru_cache(maxsize=1)
def get_vision_model() -> VisionModel:
    return VisionModel()
