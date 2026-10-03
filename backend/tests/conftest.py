import pytest

from scholargraph import agent as agent_module


@pytest.fixture(autouse=True)
def no_real_chroma_lookup(monkeypatch):
    """Keeps agent tests off the real ChromaDB.

    The reranker loads abstracts for graph-only candidates from ChromaDB. Tests that
    run the agent should never touch a real persistent store, so this returns
    nothing by default. Tests that care override it with their own monkeypatch.
    """
    monkeypatch.setattr(agent_module, "get_documents_by_ids", lambda _ids: {})
