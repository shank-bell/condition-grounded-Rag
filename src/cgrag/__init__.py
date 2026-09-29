"""Condition-Grounded Scientific RAG."""


def run(question: str, history: list[dict] | None = None):
    """Answer one question through all ten stages (imported lazily: it loads the local models)."""
    from .pipeline.run import run as _run

    return _run(question, history)
