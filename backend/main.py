
# backend/main.py

import logging

from contextlib import asynccontextmanager
from threading import Lock

from fastapi import FastAPI

from backend.config import (
    APP_NAME,
    APP_VERSION,
)

from backend.routes.chat import router
from miniai.inferencia import cargar_modelo


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):

    logger.info("Iniciando MiniAI...")

    modelo, checkpoint, device = cargar_modelo()

    app.state.modelo = modelo
    app.state.checkpoint = checkpoint
    app.state.device = device
    app.state.inference_lock = Lock()

    logger.info("Modelo cargado correctamente")
    logger.info("Dispositivo: %s", device)
    logger.info(
        "Vocabulario: %s tokens",
        len(checkpoint["vocabulario"])
    )

    yield

    logger.info("Deteniendo MiniAI...")


app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    lifespan=lifespan
)

app.include_router(router)
