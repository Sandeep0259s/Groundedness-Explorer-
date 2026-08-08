from collections.abc import Iterator

import ollama

from . import model_prefs
from .config import settings

SYSTEM_PROMPT = (
    "You are a helpful assistant for a document Q&A tool. The user's message may be a genuine "
    "question about the provided context, or just a greeting or casual remark ('hi', 'thanks', "
    "'how are you') — reply naturally and briefly to the latter, with no need to reference the "
    "context at all. For an actual question about the documents, answer using ONLY the provided "
    "context, and if the context doesn't contain the answer, say you don't know instead of "
    "guessing. Answer in plain, direct sentences — do not use bullet points, headings, or "
    "preambles like 'According to the context'."
)


class OllamaLLM:
    def __init__(self, host: str = settings.ollama_host):
        self.client = ollama.Client(host=host)

    def _build_messages(self, question: str, context_chunks: list[str], history: list[dict] | None) -> list[dict]:
        context = "\n\n".join(f"[{i+1}] {chunk}" for i, chunk in enumerate(context_chunks))
        prompt = f"Context:\n{context}\n\nUser message: {question}"

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": prompt})
        return messages

    def _active_model(self, model: str | None) -> str:
        # No model stored on self — resolved fresh so a switch made via the
        # Model panel takes effect on the very next question, no reload.
        return model or model_prefs.load_active_model("chat", settings.ollama_model)

    def generate(
        self,
        question: str,
        context_chunks: list[str],
        history: list[dict] | None = None,
        model: str | None = None,
    ) -> str:
        messages = self._build_messages(question, context_chunks, history)
        response = self.client.chat(model=self._active_model(model), messages=messages)
        return response["message"]["content"]

    def generate_stream(
        self,
        question: str,
        context_chunks: list[str],
        history: list[dict] | None = None,
        model: str | None = None,
    ) -> Iterator[str]:
        messages = self._build_messages(question, context_chunks, history)
        for chunk in self.client.chat(model=self._active_model(model), messages=messages, stream=True):
            if chunk.message.content:
                yield chunk.message.content
