from fastapi.testclient import TestClient

from scholargraph import api


def test_health_endpoint() -> None:
    response = TestClient(api.app).get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_endpoint_passes_recent_history(monkeypatch) -> None:
    class StubAgent:
        def __init__(self) -> None:
            self.call = None

        def run(self, message: str, history: list[dict[str, str]]) -> str:
            self.call = (message, history)
            return "A grounded answer."

    agent = StubAgent()
    monkeypatch.setattr(api, "get_agent", lambda: agent)
    history = [{"role": "user", "content": "Earlier question"}]

    response = TestClient(api.app).post(
        "/api/chat",
        json={"message": "Follow-up question", "history": history},
    )

    assert response.status_code == 200
    assert response.json() == {"answer": "A grounded answer."}
    assert agent.call == ("Follow-up question", history)


def test_chat_endpoint_rejects_invalid_history() -> None:
    response = TestClient(api.app).post(
        "/api/chat",
        json={"message": "Question", "history": [{"role": "system", "content": "No"}]},
    )

    assert response.status_code == 422


def test_chat_endpoint_rejects_blank_message() -> None:
    response = TestClient(api.app).post("/api/chat", json={"message": "   "})

    assert response.status_code == 422
