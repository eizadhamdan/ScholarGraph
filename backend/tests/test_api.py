import pytest
from fastapi.testclient import TestClient

from scholargraph import api, chat_store


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(chat_store, "DATABASE_PATH", tmp_path / "chat-history.sqlite3")
    with TestClient(api.app) as test_client:
        yield test_client


def test_health_endpoint(client: TestClient) -> None:
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_endpoint_persists_messages_and_uses_stored_history(
    client: TestClient, monkeypatch
) -> None:
    class StubAgent:
        def __init__(self) -> None:
            self.calls = []

        def run(self, message: str, history: list[dict[str, str]]) -> str:
            self.calls.append((message, history))
            return "A grounded answer."

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
    assert result["answer"] == "A grounded answer."
    assert result["conversation"]["title"] == "First question"
    assert result["user_message"]["content"] == "First question"
    assert result["assistant_message"]["content"] == "A grounded answer."
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
            {"role": "assistant", "content": "A grounded answer."},
        ],
    )

    detail = client.get(f"/api/conversations/{conversation_id}").json()
    assert [message["role"] for message in detail["messages"]] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]


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


def test_chat_endpoint_rejects_blank_message(client: TestClient) -> None:
    created = client.post("/api/conversations").json()
    response = client.post(
        "/api/chat",
        json={"message": "   ", "conversation_id": created["id"]},
    )

    assert response.status_code == 422
