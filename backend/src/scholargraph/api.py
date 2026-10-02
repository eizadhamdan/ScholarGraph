import logging
import os
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from starlette.concurrency import run_in_threadpool

from .agent import ScholarGraphAgent
from . import chat_store

logger = logging.getLogger(__name__)


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class ConversationSummary(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str


class StoredMessage(ChatTurn):
    id: int
    created_at: str


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
def health() -> dict[str, str]:
    return {"status": "ok"}


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
    answer = get_agent().run(request.message, history)
    assistant_message = chat_store.add_assistant_message(
        request.conversation_id, answer
    )
    return ChatResponse(
        conversation=conversation,
        answer=answer,
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
    except Exception as error:
        logger.exception("ScholarGraph query failed")
        raise HTTPException(
            status_code=500,
            detail="The research query could not be completed. Check the backend logs.",
        ) from error
