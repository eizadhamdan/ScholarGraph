import threading
import time
from types import SimpleNamespace

import pytest

from scholargraph import agent as agent_module
from scholargraph import graph_store, vector_store
from scholargraph.agent import ScholarGraphAgent, reciprocal_rank_fusion


# ---------------------------------------------------------------------------
# Issue 1: Reciprocal Rank Fusion must count each paper once per list
# ---------------------------------------------------------------------------


def test_rrf_counts_each_paper_once_per_list() -> None:
    # X has three graph rows (for example, three authors); Y is found by both
    # retrievers. Y is the stronger candidate, so it must win.
    graph_ids = ["X", "X", "X", "Y"]
    vector_ids = ["Y"]

    assert reciprocal_rank_fusion(graph_ids, vector_ids) == ["Y", "X"]
    assert reciprocal_rank_fusion(graph_ids, vector_ids) == reciprocal_rank_fusion(
        list(dict.fromkeys(graph_ids)), vector_ids
    )


def test_rrf_ignores_repeats_in_the_vector_list_too() -> None:
    assert reciprocal_rank_fusion(["Y"], ["X", "X", "X", "Y"]) == ["Y", "X"]


def test_rrf_still_ranks_unique_lists_by_combined_score() -> None:
    assert reciprocal_rank_fusion(["A", "B"], ["B", "C"]) == ["B", "A", "C"]
    assert reciprocal_rank_fusion(["A", "B", "C"], [], top_n=2) == ["A", "B"]


def test_candidate_fusion_ignores_repeated_graph_rows() -> None:
    agent = ScholarGraphAgent.__new__(ScholarGraphAgent)
    graph_rows = [
        {"paper_id": "X", "author": "Author 1"},
        {"paper_id": "X", "author": "Author 2"},
        {"paper_id": "X", "author": "Author 3"},
        {"paper_id": "Y", "author": "Author 4"},
    ]
    vector_hits = [{"paper_id": "Y", "document": "Abstract.", "metadata": {}}]

    result = agent._node_candidate_fusion(
        {"graph_raw_results": graph_rows, "vector_raw_results": vector_hits}
    )

    assert result["fused_paper_ids"] == ["Y", "X"]


# ---------------------------------------------------------------------------
# Issue 2: graph-only candidates must get their abstract before reranking
# ---------------------------------------------------------------------------


def _reranker_state() -> dict:
    return {
        "user_query": "graph methods",
        "fused_paper_ids": ["W-graph", "W-vector"],
        "vector_raw_results": [
            {
                "paper_id": "W-vector",
                "document": "Vector abstract.",
                "metadata": {"title": "Vector paper", "published": "2025"},
            }
        ],
        "expanded_graph_context": {
            "W-graph": {
                "title": "Graph paper",
                "authors": ["A. Author"],
                "concepts": ["GNN"],
                "categories": ["cs.LG"],
            },
            "W-vector": {
                "title": "Vector paper",
                "authors": [],
                "concepts": [],
                "categories": [],
            },
        },
    }


def _agent_with_recording_reranker():
    scored_pairs = []

    def predict(pairs):
        scored_pairs.extend(pairs)
        return [0.5 for _ in pairs]

    agent = ScholarGraphAgent.__new__(ScholarGraphAgent)
    agent.reranker = SimpleNamespace(predict=predict)
    return agent, scored_pairs


def _payload(result: dict, paper_id: str) -> dict:
    return next(p for p in result["reranked_evidence"] if p["paper_id"] == paper_id)


def test_reranker_loads_abstracts_for_graph_only_candidates(monkeypatch) -> None:
    lookups = []

    def fake_lookup(paper_ids):
        lookups.append(list(paper_ids))
        return {"W-graph": {"document": "Stored abstract.", "metadata": {}}}

    monkeypatch.setattr(agent_module, "get_documents_by_ids", fake_lookup)
    agent, scored_pairs = _agent_with_recording_reranker()

    result = agent._node_reranker(_reranker_state())

    assert lookups == [["W-graph"]]  # the vector hit is not looked up again
    assert _payload(result, "W-graph")["abstract"] == "Stored abstract."
    assert _payload(result, "W-vector")["abstract"] == "Vector abstract."
    assert any("Abstract: Stored abstract." in pair[1] for pair in scored_pairs)


def test_reranker_skips_lookup_when_every_candidate_has_an_abstract(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        agent_module,
        "get_documents_by_ids",
        lambda _ids: pytest.fail("no lookup is needed"),
    )
    agent, _ = _agent_with_recording_reranker()
    state = _reranker_state()
    state["fused_paper_ids"] = ["W-vector"]

    result = agent._node_reranker(state)

    assert _payload(result, "W-vector")["abstract"] == "Vector abstract."


def test_reranker_still_works_when_the_abstract_lookup_fails(monkeypatch) -> None:
    def failing_lookup(_ids):
        raise RuntimeError("ChromaDB is unavailable")

    monkeypatch.setattr(agent_module, "get_documents_by_ids", failing_lookup)
    agent, _ = _agent_with_recording_reranker()

    result = agent._node_reranker(_reranker_state())

    assert _payload(result, "W-graph")["abstract"] == ""
    assert _payload(result, "W-vector")["abstract"] == "Vector abstract."
    assert result["source_ids"] == {"W-graph", "W-vector"}


def test_get_documents_by_ids_returns_stored_abstracts(monkeypatch) -> None:
    calls = []

    class FakeCollection:
        def get(self, **kwargs):
            calls.append(kwargs)
            # Chroma omits IDs it does not know, so W-missing is not returned.
            return {
                "ids": ["W-1", "W-2"],
                "documents": ["Abstract one.", None],
                "metadatas": [{"title": "One"}, None],
            }

    fake_client = SimpleNamespace(get_collection=lambda name: FakeCollection())
    monkeypatch.setattr(vector_store, "get_chroma_client", lambda: fake_client)

    found = vector_store.get_documents_by_ids(["W-1", "W-2", "W-missing"])

    assert found == {
        "W-1": {"document": "Abstract one.", "metadata": {"title": "One"}},
        "W-2": {"document": "", "metadata": {}},
    }
    assert calls[0]["ids"] == ["W-1", "W-2", "W-missing"]


def test_get_documents_by_ids_with_no_ids_does_not_touch_chroma(monkeypatch) -> None:
    monkeypatch.setattr(
        vector_store,
        "get_chroma_client",
        lambda: pytest.fail("Chroma must not be opened for an empty lookup"),
    )

    assert vector_store.get_documents_by_ids([]) == {}


# ---------------------------------------------------------------------------
# Issue 9: one shared Neo4j driver instead of one per call
# ---------------------------------------------------------------------------


class _FakeTransaction:
    def run(self, _cypher, _parameters=None):
        return [SimpleNamespace(data=lambda: {"paper_id": "W1"})]


class _FakeSession:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def execute_read(self, operation):
        return operation(_FakeTransaction())


class _FakeDriver:
    def __init__(self, verify_error: Exception | None = None) -> None:
        self.verify_error = verify_error
        self.closed = 0

    def session(self):
        return _FakeSession()

    def verify_connectivity(self) -> None:
        if self.verify_error is not None:
            raise self.verify_error

    def close(self) -> None:
        self.closed += 1


@pytest.fixture
def fake_neo4j(monkeypatch):
    created: list[_FakeDriver] = []

    def make_driver(*_args, **_kwargs):
        time.sleep(0.01)  # widens the race window for the threading test
        driver = _FakeDriver()
        created.append(driver)
        return driver

    monkeypatch.setattr(graph_store, "GraphDatabase", SimpleNamespace(driver=make_driver))
    monkeypatch.setattr(graph_store, "_driver", None)
    yield created
    graph_store.close_neo4j_driver()


def test_neo4j_driver_is_created_once_and_reused(fake_neo4j) -> None:
    assert graph_store.check_graph_connection() is True
    assert graph_store.check_graph_connection() is True
    assert graph_store.run_cypher_query("RETURN 1") == [{"paper_id": "W1"}]
    assert graph_store.run_cypher_query("RETURN 1") == [{"paper_id": "W1"}]

    assert len(fake_neo4j) == 1
    assert fake_neo4j[0].closed == 0  # queries and health checks never close it


def test_close_neo4j_driver_closes_it_and_the_next_call_reconnects(fake_neo4j) -> None:
    graph_store.check_graph_connection()
    graph_store.close_neo4j_driver()
    assert fake_neo4j[0].closed == 1

    graph_store.close_neo4j_driver()  # closing twice is harmless
    assert fake_neo4j[0].closed == 1

    graph_store.check_graph_connection()
    assert len(fake_neo4j) == 2


def test_concurrent_first_use_creates_a_single_driver(fake_neo4j) -> None:
    drivers = []
    barrier = threading.Barrier(8)

    def worker() -> None:
        barrier.wait()
        drivers.append(graph_store.get_neo4j_driver())

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(fake_neo4j) == 1
    assert all(driver is fake_neo4j[0] for driver in drivers)


def test_failed_health_check_reports_false_and_keeps_the_driver(
    monkeypatch, fake_neo4j
) -> None:
    graph_store.get_neo4j_driver().verify_error = OSError("Neo4j is offline")

    assert graph_store.check_graph_connection() is False
    assert fake_neo4j[0].closed == 0
