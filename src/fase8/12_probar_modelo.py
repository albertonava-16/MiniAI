"""Verifica que modelo.py carga exactamente el checkpoint de Fase 7."""
from pathlib import Path

import torch

from modelo import MiniGPT


def main():
    ruta_checkpoint = Path(__file__).resolve().parent.parent / "fase7" / "miniai_fase7_checkpoint.pth"
    if not ruta_checkpoint.is_file():
        raise FileNotFoundError(f"No se encontró el checkpoint: {ruta_checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(ruta_checkpoint, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    vocabulario = checkpoint["vocabulario"]
    modelo = MiniGPT(
        vocab_size=len(vocabulario),
        embedding_dim=config["embedding_dim"],
        num_heads=config["num_heads"],
        max_seq_len=config["max_seq_len"],
        num_blocks=config["num_blocks"],
        dropout=config["dropout"],
    )

    # strict=True verifica las claves y las formas esperadas del checkpoint.
    modelo.load_state_dict(checkpoint["model_state_dict"], strict=True)
    modelo = modelo.to(device)
    modelo.eval()

    bos = checkpoint["special_tokens"]["BOS"]
    entrada = torch.tensor([[vocabulario.index(bos)]], dtype=torch.long, device=device)
    with torch.inference_mode():
        logits = modelo(entrada)

    esperado = (1, 1, len(vocabulario))
    assert tuple(logits.shape) == esperado, (
        f"Forma inesperada: {tuple(logits.shape)}, se esperaba {esperado}"
    )
    assert torch.isfinite(logits).all().item(), "Se encontraron logits no finitos"

    print("Dispositivo:", device)
    print("Checkpoint:", ruta_checkpoint.name)
    print("Vocabulario:", len(vocabulario), "tokens")
    print("Arquitectura compatible: OK")
    print("Forma de logits:", tuple(logits.shape))
    print("Forward sin entrenamiento: OK")
    print("PRUEBA COMPLETADA")


if __name__ == "__main__":
    main()
