
# backend/routes/chat.py

import logging
import time
import torch

from fastapi import APIRouter, HTTPException, Request

from backend.schemas.chat import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
)

from miniai.generacion import generar


router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/health", response_model=HealthResponse)
def health(request: Request):

    device = request.app.state.device

    return HealthResponse(
        status="ok",
        modelo="MiniGPT - Fase 7",
        dispositivo=str(device)
    )


@router.post("/chat", response_model=ChatResponse)
def chat(datos: ChatRequest, request: Request):

    modelo = request.app.state.modelo
    checkpoint = request.app.state.checkpoint
    device = request.app.state.device
    lock = request.app.state.inference_lock

    try:
        # Evita inferencias simultáneas mientras
        # estudiamos la concurrencia del servidor.
        with lock:

            if device.type == "cuda":
                torch.cuda.synchronize(device)

            inicio = time.perf_counter()

            respuesta = generar(
                modelo=modelo,
                checkpoint=checkpoint,
                device=device,
                prompt=datos.prompt,
                max_tokens=datos.max_tokens,
                temperature=datos.temperature,
                top_k=datos.top_k,
                top_p=datos.top_p
            )

            if device.type == "cuda":
                torch.cuda.synchronize(device)

            latencia_ms = (
                time.perf_counter() - inicio
            ) * 1000

    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail=str(error)
        ) from error

    except Exception:
        logger.exception("Error interno durante inferencia")
        raise HTTPException(
            status_code=500,
            detail="Error interno de MiniAI"
        )

    logger.info(
        "Generación completada | %.2f ms | %s",
        latencia_ms,
        device
    )

    return ChatResponse(
        prompt=datos.prompt,
        respuesta=respuesta,
        dispositivo=str(device),
        latencia_ms=round(latencia_ms, 2)
    )
