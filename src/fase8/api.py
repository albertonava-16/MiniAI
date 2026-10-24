"""MiniAI Fase 8: API HTTP local para inferencia.

Ejecutar desde src/fase8:
    py -m uvicorn api:app --host 127.0.0.1 --port 8000
"""
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from generar import cargar_miniai, generar

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("miniai")

# Se cargará una sola vez al iniciar el servidor.
estado = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    modelo, checkpoint, device = cargar_miniai()
    estado.update(modelo=modelo, checkpoint=checkpoint, device=device)
    logger.info("MiniAI cargado en %s (%d tokens)", device, len(checkpoint["vocabulario"]))
    yield
    estado.clear()


app = FastAPI(title="MiniAI API", version="0.1.0", lifespan=lifespan)


class ChatRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=200)
    max_tokens: int = Field(default=12, ge=1, le=50)
    temperature: float = Field(default=1.0, gt=0, le=2)
    top_k: int | None = Field(default=None, ge=1)
    top_p: float = Field(default=0.9, gt=0, le=1)

class ChatResponse(BaseModel):
    prompt: str
    respuesta: str
    dispositivo: str
    latencia_ms: float


@app.get("/health")
def health():
    return {"status": "ok", "modelo_cargado": bool(estado)}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if not estado:
        raise HTTPException(status_code=503, detail="Modelo no cargado")

    inicio = time.perf_counter()
    try:
        respuesta = generar(
            estado["modelo"], estado["checkpoint"], estado["device"],
            request.prompt, request.max_tokens, request.temperature,
            request.top_k, request.top_p,
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    # Las operaciones CUDA pueden ser asíncronas; sincronizar para medir latencia real.
    if estado["device"].type == "cuda":
        import torch
        torch.cuda.synchronize(estado["device"])

    latencia_ms = round((time.perf_counter() - inicio) * 1000, 2)
    logger.info("/chat prompt=%r latencia_ms=%.2f", request.prompt, latencia_ms)
    return ChatResponse(
        prompt=request.prompt,
        respuesta=respuesta,
        dispositivo=str(estado["device"]),
        latencia_ms=latencia_ms,
    )
