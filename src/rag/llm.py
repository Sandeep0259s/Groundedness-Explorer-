import ollama

from . import model_prefs
from .config import settings

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using ONLY the provided context. "
    "If the context does not contain the answer, say you don't know instead of guessing. "
    "Answer in plain, direct sentences that state facts on their own — do not use bullet points, "
    "headings, or preambles like 'According to the context'."
)


class OllamaLLM:
    def __init__(self, host: str = settings.ollama_host):
        self.client = ollama.Client(host=host)

    def generate(
        self,
        question: str,
        context_chunks: list[str],
        history: list[dict] | None = None,
        model: str | None = None,
    ) -> str:
        context = "\n\n".join(f"[{i+1}] {chunk}" for i, chunk in enumerate(context_chunks))
        prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer using only the context above:"

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": prompt})

        # No model stored on self — resolved fresh so a switch made via the
        # Model panel takes effect on the very next question, no reload.
        active = model or model_prefs.load_active_model("chat", settings.ollama_model)
        response = self.client.chat(model=active, messages=messages)
        return response["message"]["content"]
