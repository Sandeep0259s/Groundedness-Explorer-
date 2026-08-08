"""Rewrites a conversational follow-up into a standalone question before
retrieval, using the conversation history for context.

Multi-turn generation already works without this — the chat history is
passed straight to the LLM, which has no trouble answering "which continent
is it on?" once it can see the prior turn. Retrieval is the part that never
sees conversation history at all: it embeds/BM25-searches the raw follow-up
text alone, so "which continent is it on?" retrieves on "continent" and
"it" with no signal that "it" means the Eiffel Tower. This fixes retrieval
specifically, without changing what generation sees — the final answer step
still gets the original question plus the full history, unchanged.
"""
from . import ollama_client

_REWRITE_PROMPT = """Conversation so far:
{history}

Follow-up question: {question}

Rewrite the follow-up question as a standalone question that includes
whatever context from the conversation it depends on, so it can be
understood with no prior context. If it's already standalone, repeat it
unchanged. Reply with ONLY the rewritten question — no preamble, no quotes."""


def rewrite_for_retrieval(question: str, history: list[dict] | None, model: str | None = None) -> str:
    """Returns `question` unchanged if there's no history, if the rewrite
    call fails for any reason, or if the model returns nothing usable — a
    rewrite failure should never block answering the actual question."""
    if not history:
        return question

    transcript = "\n".join(f"{turn['role']}: {turn['content']}" for turn in history)
    prompt = _REWRITE_PROMPT.format(history=transcript, question=question)

    try:
        response = ollama_client.get_client().chat(
            model=ollama_client.active_chat_model(model),
            messages=[{"role": "user", "content": prompt}],
            think=False,
        )
        rewritten = response["message"]["content"].strip().strip('"')
        return rewritten if rewritten else question
    except Exception:
        return question
