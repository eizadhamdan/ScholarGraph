import logging
import os
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from starlette.concurrency import run_in_threadpool

from .agent import (
    AgentResult,
    GraphQueryError,
    GroundingValidationError,
    InsufficientEvidenceError,
    ModelUnavailableError,
    ScholarGraphAgent,
)
from . import chat_store
from .graph_store import GraphUnavailableError, check_graph_connection

logger = logging.getLogger(__name__)


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class ConversationSummary(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str


class VectorSource(BaseModel):
    paper_id: str
    title: str | None = None
    published: str | None = None
    excerpt: str


class GraphQueryAttempt(BaseModel):
    kind: str
    cypher: str
    result_count: int
    parameters: dict[str, Any] | None = None
    error: str | None = None


class RetrievalTrace(BaseModel):
    graph_queries: list[GraphQueryAttempt]
    graph_result_count: int
    graph_records: list[dict[str, Any]]
    graph_results_truncated: bool
    vector_hit_count: int
    vector_sources: list[VectorSource]


class StoredMessage(ChatTurn):
    id: int
    created_at: str
    retrieval: RetrievalTrace | None = None


class ConversationDetail(ConversationSummary):
    messages: list[StoredMessage]


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    conversation_id: str = Field(min_length=1, max_length=64)

    @field_validator("message")
    @classmethod
    def message_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Message cannot be blank.")
        return value


class ChatResponse(BaseModel):
    conversation: ConversationSummary
    answer: str
    retrieval: RetrievalTrace
    user_message: StoredMessage
    assistant_message: StoredMessage


@lru_cache(maxsize=1)
def get_agent() -> ScholarGraphAgent:
    return ScholarGraphAgent()


@asynccontextmanager
async def lifespan(_: FastAPI):
    chat_store.initialize_database()
    yield


app = FastAPI(title="ScholarGraph API", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        os.getenv("FRONTEND_ORIGIN", "http://localhost:5173"),
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type"],
)


@app.get("/api/health")
def health() -> dict[str, str | bool]:
    graph_available = check_graph_connection()
    return {
        "status": "ok" if graph_available else "degraded",
        "knowledge_graph_available": graph_available,
    }


@app.get("/api/conversations", response_model=list[ConversationSummary])
def list_conversations() -> list[dict]:
    return chat_store.list_conversations()


@app.post(
    "/api/conversations",
    response_model=ConversationSummary,
    status_code=201,
)
def create_conversation() -> dict:
    return chat_store.create_conversation()


@app.get("/api/conversations/{conversation_id}", response_model=ConversationDetail)
def get_conversation(conversation_id: str) -> dict:
    conversation = chat_store.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return conversation


@app.delete("/api/conversations/{conversation_id}", status_code=204)
def delete_conversation(conversation_id: str) -> None:
    if not chat_store.delete_conversation(conversation_id):
        raise HTTPException(status_code=404, detail="Conversation not found.")


def _run_chat(request: ChatRequest) -> ChatResponse:
    stored = chat_store.add_user_message(request.conversation_id, request.message)
    if stored is None:
        raise chat_store.ConversationNotFoundError(request.conversation_id)

    conversation, user_message, history = stored
    result: AgentResult = get_agent().run_with_evidence(request.message, history)
    assistant_message = chat_store.add_assistant_message(
        request.conversation_id, result.answer, result.retrieval
    )
    return ChatResponse(
        conversation=conversation,
        answer=result.answer,
        retrieval=result.retrieval,
        user_message=user_message,
        assistant_message=assistant_message,
    )


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    try:
        return await run_in_threadpool(_run_chat, request)
    except chat_store.ConversationNotFoundError as error:
        raise HTTPException(
            status_code=404, detail="Conversation not found."
        ) from error
    except GraphUnavailableError as error:
        logger.exception("Knowledge graph query failed; refusing to generate answer")
        raise HTTPException(
            status_code=503,
            detail=(
                "The knowledge graph is unavailable or could not be queried. "
                "No answer was generated. Check that Neo4j is running and reachable."
            ),
        ) from error
    except GraphQueryError as error:
        logger.warning("Generated Cypher was rejected: %s", error.__cause__ or error)
        raise HTTPException(
            status_code=422,
            detail=(
                "Neo4j rejected the generated graph query and the fallback search "
                "could not complete. No answer was generated."
            ),
        ) from error
    except ModelUnavailableError as error:
        logger.warning("Gemini is unavailable: %s", error.__cause__ or error)
        detail = (
            "Gemini is temporarily unavailable. No answer was generated; "
            "please retry when the model service is available."
        )
        if error.retrieval is not None:
            graph_counts = ", ".join(
                f"{attempt['kind']}: {attempt['result_count']} rows"
                for attempt in error.retrieval["graph_queries"]
            )
            detail += (
                f" Retrieval completed ({graph_counts}; "
                f"ChromaDB: {error.retrieval['vector_hit_count']} papers)."
            )
        raise HTTPException(
            status_code=503,
            detail=detail,
        ) from error
    except InsufficientEvidenceError as error:
        raise HTTPException(
            status_code=422,
            detail=(
                "No citable evidence was retrieved, so no answer was generated. "
                "Try a more specific question or check the indexed data."
            ),
        ) from error
    except GroundingValidationError as error:
        logger.exception("Grounded response validation failed")
        raise HTTPException(
            status_code=502,
            detail=(
                "The model did not produce a response grounded in retrieved sources. "
                "No answer was saved. Please try again."
            ),
        ) from error
    except Exception as error:
        logger.exception("ScholarGraph query failed")
        raise HTTPException(
            status_code=500,
            detail="The research query could not be completed. Check the backend logs.",
        ) from error
