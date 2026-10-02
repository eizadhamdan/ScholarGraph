from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from scholargraph import agent as agent_module
from scholargraph import api, chat_store
from scholargraph.agent import (
    AgentResult,
    GroundedClaim,
    GroundedResponse,
    GroundingValidationError,
    ModelUnavailableError,
    ScholarGraphAgent,
)
from scholargraph.graph_store import (
    GraphQueryError,
    GraphUnavailableError,
    run_cypher_query,
)


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(chat_store, "DATABASE_PATH", tmp_path / "chat-history.sqlite3")
    with TestClient(api.app) as test_client:
        yield test_client


def test_health_endpoint_reports_graph_available(
    client: TestClient, monkeypatch
) -> None:
    monkeypatch.setattr(api, "check_graph_connection", lambda: True)
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "knowledge_graph_available": True,
    }


def test_health_endpoint_reports_graph_unavailable(
    client: TestClient, monkeypatch
) -> None:
    monkeypatch.setattr(api, "check_graph_connection", lambda: False)

    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "degraded",
        "knowledge_graph_available": False,
    }


def test_chat_endpoint_persists_messages_and_uses_stored_history(
    client: TestClient, monkeypatch
) -> None:
    class StubAgent:
        def __init__(self) -> None:
            self.calls = []

        def run_with_evidence(
            self, message: str, history: list[dict[str, str]]
        ) -> AgentResult:
            self.calls.append((message, history))
            return AgentResult(
                answer="- A grounded answer. [W-source]",
                retrieval={
                    "graph_queries": [
                        {
                            "kind": "Gemini-generated",
                            "cypher": "MATCH (p:Paper) RETURN p.id AS paper_id",
                            "result_count": 1,
                        }
                    ],
                    "graph_result_count": 1,
                    "graph_records": [{"paper_id": "W-graph"}],
                    "graph_results_truncated": False,
                    "vector_hit_count": 1,
                    "vector_sources": [
                        {
                            "paper_id": "W-source",
                            "title": "Source paper",
                            "published": "2025-01-01",
                            "excerpt": "Evidence from the paper.",
                        }
                    ],
                },
            )

    agent = StubAgent()
    monkeypatch.setattr(api, "get_agent", lambda: agent)
    created = client.post("/api/conversations")
    conversation_id = created.json()["id"]

    response = client.post(
        "/api/chat",
        json={"message": "First question", "conversation_id": conversation_id},
    )

    assert response.status_code == 200
    result = response.json()
    assert result["answer"] == "- A grounded answer. [W-source]"
    assert result["retrieval"]["graph_result_count"] == 1
    assert result["retrieval"]["vector_hit_count"] == 1
    assert result["conversation"]["title"] == "First question"
    assert result["user_message"]["content"] == "First question"
    assert result["assistant_message"]["content"] == "- A grounded answer. [W-source]"
    assert agent.calls == [("First question", [])]

    follow_up = client.post(
        "/api/chat",
        json={"message": "Follow-up question", "conversation_id": conversation_id},
    )
    assert follow_up.status_code == 200
    assert agent.calls[1] == (
        "Follow-up question",
        [
            {"role": "user", "content": "First question"},
            {
                "role": "assistant",
                "content": "- A grounded answer. [W-source]",
            },
        ],
    )

    detail = client.get(f"/api/conversations/{conversation_id}").json()
    assert [message["role"] for message in detail["messages"]] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert detail["messages"][1]["retrieval"]["graph_result_count"] == 1


def test_conversation_list_and_delete(client: TestClient) -> None:
    created = client.post("/api/conversations")
    conversation_id = created.json()["id"]

    listed = client.get("/api/conversations")
    assert [item["id"] for item in listed.json()] == [conversation_id]

    deleted = client.delete(f"/api/conversations/{conversation_id}")
    assert deleted.status_code == 204
    assert client.get(f"/api/conversations/{conversation_id}").status_code == 404
    assert client.get("/api/conversations").json() == []


def test_chat_rejects_missing_conversation(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(api, "get_agent", lambda: pytest.fail("agent must not run"))
    response = client.post(
        "/api/chat",
        json={"message": "Question", "conversation_id": "missing"},
    )

    assert response.status_code == 404


def test_chat_refuses_answer_when_graph_is_unavailable(
    client: TestClient, monkeypatch
) -> None:
    class GraphUnavailableAgent:
        def run_with_evidence(
            self, message: str, history: list[dict[str, str]]
        ) -> AgentResult:
            raise GraphUnavailableError("Neo4j is offline")

    monkeypatch.setattr(api, "get_agent", lambda: GraphUnavailableAgent())
    conversation_id = client.post("/api/conversations").json()["id"]

    response = client.post(
        "/api/chat",
        json={"message": "Research question", "conversation_id": conversation_id},
    )

    assert response.status_code == 503
    assert "No answer was generated" in response.json()["detail"]
    saved = client.get(f"/api/conversations/{conversation_id}").json()
    assert [message["role"] for message in saved["messages"]] == ["user"]


def test_gemini_outage_reports_completed_retrieval(
    client: TestClient, monkeypatch
) -> None:
    class GeminiUnavailableAgent:
        def run_with_evidence(self, message: str, history: list[dict[str, str]]):
            raise ModelUnavailableError(
                "Gemini is unavailable during synthesis.",
                retrieval={
                    "graph_queries": [
                        {"kind": "Gemini-generated", "result_count": 0},
                        {"kind": "keyword fallback", "result_count": 2},
                    ],
                    "vector_hit_count": 5,
                },
            )

    monkeypatch.setattr(api, "get_agent", lambda: GeminiUnavailableAgent())
    conversation_id = client.post("/api/conversations").json()["id"]
    response = client.post(
        "/api/chat",
        json={"message": "Research question", "conversation_id": conversation_id},
    )

    assert response.status_code == 503
    assert "Gemini is temporarily unavailable" in response.json()["detail"]
    assert "keyword fallback: 2 rows" in response.json()["detail"]
    assert "ChromaDB: 5 papers" in response.json()["detail"]


def test_graph_query_failure_is_not_returned_as_result_data(monkeypatch) -> None:
    class FailedSession:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def execute_read(self, operation):
            return operation(self)

        def run(self, _query: str, _parameters: dict | None = None):
            raise OSError("Neo4j is offline")

    class FailedDriver:
        def session(self):
            return FailedSession()

        def close(self) -> None:
            pass

    monkeypatch.setattr(
        "scholargraph.graph_store.get_neo4j_driver", lambda: FailedDriver()
    )

    with pytest.raises(GraphUnavailableError):
        run_cypher_query("MATCH (p:Paper) RETURN p")


def test_invalid_cypher_is_distinguished_from_connection_failure(monkeypatch) -> None:
    class SyntaxFailure(Exception):
        code = "Neo.ClientError.Statement.SyntaxError"

    class InvalidSession:
        def execute_read(self, operation):
            return operation(self)

        def run(self, _query: str, _parameters: dict | None = None):
            raise SyntaxFailure("invalid Cypher")

    class InvalidDriver:
        def session(self):
            return InvalidSession()

        def close(self) -> None:
            pass

    monkeypatch.setattr(
        "scholargraph.graph_store.get_neo4j_driver", lambda: InvalidDriver()
    )

    with pytest.raises(GraphQueryError):
        run_cypher_query("INVALID CYPHER")


def test_agent_stops_before_vector_retrieval_on_graph_failure(monkeypatch) -> None:
    class Models:
        calls = 0

        def generate_content(self, **_kwargs):
            self.calls += 1
            return SimpleNamespace(text="MATCH (p:Paper) RETURN p")

    models = Models()
    agent = ScholarGraphAgent.__new__(ScholarGraphAgent)
    agent.client = SimpleNamespace(models=models)
    monkeypatch.setattr(agent_module, "check_graph_connection", lambda: True)

    def fail_graph_query(_cypher: str):
        raise GraphUnavailableError("Neo4j is offline")

    monkeypatch.setattr(agent_module, "run_cypher_query", fail_graph_query)
    monkeypatch.setattr(
        agent_module,
        "query_vector_store",
        lambda *_args, **_kwargs: pytest.fail("vector search must not run"),
    )

    with pytest.raises(GraphUnavailableError):
        agent.run("Find papers about graph retrieval")

    assert models.calls == 1


def test_agent_refuses_before_cypher_generation_when_graph_is_offline(
    monkeypatch,
) -> None:
    class Models:
        def generate_content(self, **_kwargs):
            pytest.fail("Gemini must not be called while Neo4j is offline")

    agent = ScholarGraphAgent.__new__(ScholarGraphAgent)
    agent.client = SimpleNamespace(models=Models())
    monkeypatch.setattr(agent_module, "check_graph_connection", lambda: False)

    with pytest.raises(GraphUnavailableError):
        agent.run("Find papers about graph retrieval")


def test_agent_runs_graph_then_vector_then_grounded_synthesis(monkeypatch) -> None:
    events = []

    class Models:
        def generate_content(self, **_kwargs):
            events.append("synthesis")
            response = GroundedResponse(
                claims=[
                    GroundedClaim(
                        statement="RAG combines retrieval with generation.",
                        source_ids=["W-vector"],
                    )
                ]
            )
            return SimpleNamespace(text=response.model_dump_json())

    agent = ScholarGraphAgent.__new__(ScholarGraphAgent)
    agent.client = SimpleNamespace(models=Models())
    agent._generate_cypher = lambda _query: "MATCH (p:Paper) RETURN p.id AS paper_id"
    monkeypatch.setattr(agent_module, "check_graph_connection", lambda: True)

    def graph_search(_cypher: str, parameters: dict | None = None):
        events.append("graph")
        if parameters is not None:
            events[-1] = "graph_fallback"
            assert "retrieval" in parameters["terms"]
            return [{"paper_id": "W-graph", "concept": "retrieval generation"}]
        return []

    def vector_search(_query: str, n_results: int):
        events.append("vector")
        assert n_results == 5
        return [
            {
                "paper_id": "W-vector",
                "document": "RAG combines retrieval with generation.",
                "metadata": {"title": "A RAG paper", "published": "2025"},
            }
        ]

    monkeypatch.setattr(agent_module, "run_cypher_query", graph_search)
    monkeypatch.setattr(agent_module, "query_vector_store", vector_search)

    result = agent.run_with_evidence("What is RAG?")

    assert events == ["graph", "graph_fallback", "vector", "synthesis"]
    assert result.answer == "- RAG combines retrieval with generation. [W-vector]"
    assert result.retrieval["graph_result_count"] == 1
    assert [attempt["kind"] for attempt in result.retrieval["graph_queries"]] == [
        "Gemini-generated",
        "keyword fallback",
    ]
    assert result.retrieval["vector_hit_count"] == 1


def test_agent_rejects_citations_not_in_retrieved_sources(monkeypatch) -> None:
    class Models:
        def generate_content(self, **_kwargs):
            response = GroundedResponse(
                claims=[GroundedClaim(statement="Unsupported", source_ids=["W-fake"])]
            )
            return SimpleNamespace(text=response.model_dump_json())

    agent = ScholarGraphAgent.__new__(ScholarGraphAgent)
    agent.client = SimpleNamespace(models=Models())
    agent._generate_cypher = lambda _query: "MATCH (p:Paper) RETURN p.id AS paper_id"
    monkeypatch.setattr(agent_module, "check_graph_connection", lambda: True)
    monkeypatch.setattr(
        agent_module,
        "run_cypher_query",
        lambda _query, _parameters=None: [],
    )
    monkeypatch.setattr(
        agent_module,
        "query_vector_store",
        lambda *_args, **_kwargs: [
            {
                "paper_id": "W-real",
                "document": "Evidence.",
                "metadata": {"title": "Paper"},
            }
        ],
    )

    with pytest.raises(GroundingValidationError):
        agent.run_with_evidence("Question")


def test_chat_endpoint_rejects_blank_message(client: TestClient) -> None:
    created = client.post("/api/conversations").json()
    response = client.post(
        "/api/chat",
        json={"message": "   ", "conversation_id": created["id"]},
    )

    assert response.status_code == 422
