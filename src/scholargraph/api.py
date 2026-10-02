import logging
import os
from functools import lru_cache
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from starlette.concurrency import run_in_threadpool

from .agent import ScholarGraphAgent

logger = logging.getLogger(__name__)


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    history: list[ChatTurn] = Field(default_factory=list, max_length=20)

    @field_validator("message")
    @classmethod
    def message_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Message cannot be blank.")
        return value


class ChatResponse(BaseModel):
    answer: str


@lru_cache(maxsize=1)
def get_agent() -> ScholarGraphAgent:
    return ScholarGraphAgent()


app = FastAPI(title="ScholarGraph API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        os.getenv("FRONTEND_ORIGIN", "http://localhost:5173"),
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    try:
        answer = await run_in_threadpool(
            get_agent().run,
            request.message.strip(),
            [turn.model_dump() for turn in request.history],
        )
    except Exception as error:
        logger.exception("ScholarGraph query failed")
        raise HTTPException(
            status_code=500,
            detail="The research query could not be completed. Check the backend logs.",
        ) from error

    return ChatResponse(answer=answer)
