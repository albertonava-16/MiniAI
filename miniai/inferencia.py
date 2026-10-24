from pathlib import Path

import torch

from miniai.modelo import MiniGPT


# Raíz del repositorio MiniAI
BASE_DIR = Path(__file__).resolve().parent.parent

# Ubicación centralizada del checkpoint
CHECKPOINT_PATH = (
    BASE_DIR / "models" / "miniai_fase7_checkpoint.pth"
)


def cargar_modelo():
    """
    Reconstruye MiniGPT utilizando la configuración
    y los pesos guardados durante la Fase 7.
    """

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=True
    )

    config = checkpoint["config"]
    vocabulario = checkpoint["vocabulario"]

    modelo = MiniGPT(
        vocab_size=len(vocabulario),
        embedding_dim=config["embedding_dim"],
        num_heads=config["num_heads"],
        max_seq_len=config["max_seq_len"],
        num_blocks=config["num_blocks"],
        dropout=config["dropout"]
    ).to(device)

    # Verificamos compatibilidad exacta de arquitectura
    modelo.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True
    )

    # Desactivamos dropout durante inferencia
    modelo.eval()

    return modelo, checkpoint, device