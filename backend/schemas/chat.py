
# backend/schemas/chat.py

from pydantic import BaseModel, Field

from backend.config import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
)


class ChatRequest(BaseModel):
    prompt: str = Field(
        min_length=1,
        max_length=500
    )

    max_tokens: int = Field(
        default=DEFAULT_MAX_TOKENS,
        ge=1,
        le=64
    )

    temperature: float = Field(
        default=DEFAULT_TEMPERATURE,
        gt=0,
        le=2
    )

    top_k: int | None = Field(
        default=DEFAULT_TOP_K,
        ge=1
    )

    top_p: float = Field(
        default=DEFAULT_TOP_P,
        gt=0,
        le=1
    )


class ChatResponse(BaseModel):
    prompt: str
    respuesta: str
    dispositivo: str
    latencia_ms: float


class HealthResponse(BaseModel):
    status: str
    modelo: str
    dispositivo: str
