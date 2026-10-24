"""Fase 8: inferencia autónoma de MiniGPT usando el checkpoint de Fase 7.

Ejemplo:
    py generar.py "el perro" --max-tokens 12 --temperature 1 --top-p 0.9
"""
import argparse
from pathlib import Path

import torch

from modelo import MiniGPT


RUTA_CHECKPOINT = Path(__file__).resolve().parent.parent / "fase7" / "miniai_fase7_checkpoint.pth"


def cargar_miniai(ruta_checkpoint=RUTA_CHECKPOINT):
    """Reconstruye el modelo a partir del checkpoint; no entrena."""
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
    modelo.load_state_dict(checkpoint["model_state_dict"], strict=True)
    modelo.to(device)
    modelo.eval()
    return modelo, checkpoint, device


def seleccionar_token(logits, temperature=1.0, top_k=None, top_p=0.9):
    """Muestrea un token desde logits 1D aplicando temperature, top-k y top-p."""
    if temperature <= 0:
        raise ValueError("temperature debe ser mayor que 0")
    if top_k is not None and top_k < 1:
        raise ValueError("top_k debe ser un entero positivo")
    if not (0 < top_p <= 1):
        raise ValueError("top_p debe estar entre 0 (excluido) y 1")

    logits = logits / temperature
    if top_k is not None:
        k = min(top_k, logits.numel())
        minimo = torch.topk(logits, k).values[-1]
        logits = logits.masked_fill(logits < minimo, float("-inf"))

    probs = torch.softmax(logits, dim=-1)
    probs_ordenadas, indices_ordenados = torch.sort(probs, descending=True)
    acumuladas = torch.cumsum(probs_ordenadas, dim=-1)

    # Conservar el primer token que cruza el umbral.
    eliminar = acumuladas > top_p
    eliminar[1:] = eliminar[:-1].clone()
    eliminar[0] = False

    probs_ordenadas = probs_ordenadas.masked_fill(eliminar, 0)
    probs_ordenadas = probs_ordenadas / probs_ordenadas.sum()
    indice_ordenado = torch.multinomial(probs_ordenadas, num_samples=1).item()
    return indices_ordenados[indice_ordenado].item()

def generar(modelo, checkpoint, device, prompt, max_tokens=12, temperature=1.0,
            top_k=None, top_p=0.9):
    """Genera una continuación para un prompt formado por palabras del vocabulario."""
    if max_tokens < 1:
        raise ValueError("max_tokens debe ser positivo")
    vocabulario = checkpoint["vocabulario"]
    token_a_id = {token: i for i, token in enumerate(vocabulario)}
    especiales = checkpoint["special_tokens"]
    bos, eos = especiales["BOS"], especiales["EOS"]
    tokens_prompt = prompt.strip().lower().split()
    if not tokens_prompt:
        raise ValueError("El prompt no puede estar vacío")
    desconocidos = [token for token in tokens_prompt if token not in token_a_id]
    if desconocidos:
        raise ValueError(f"Tokens desconocidos: {desconocidos}. Usa palabras del corpus de Fase 7.")
    if bos in tokens_prompt or eos in tokens_prompt:
        raise ValueError("No introduzcas BOS ni EOS manualmente")

    ids = [token_a_id[bos]] + [token_a_id[token] for token in tokens_prompt]
    max_contexto = checkpoint["config"]["max_seq_len"]
    generados = []
    with torch.inference_mode():
        for _ in range(max_tokens):
            contexto = ids[-max_contexto:]
            contexto = [token_a_id[bos]] * (max_contexto - len(contexto)) + contexto
            entrada = torch.tensor([contexto], dtype=torch.long, device=device)
            logits = modelo(entrada)[0, -1, :].clone()
            logits[token_a_id[bos]] = float("-inf")  # BOS solo indica inicio.
            siguiente = seleccionar_token(logits, temperature, top_k, top_p)
            if siguiente == token_a_id[eos]:
                break
            ids.append(siguiente)
            generados.append(vocabulario[siguiente])
    return " ".join(tokens_prompt + generados)


def main():
    parser = argparse.ArgumentParser(description="Generar texto con MiniAI (Fase 8)")
    parser.add_argument("prompt", nargs="?", default="el perro")
    parser.add_argument("--max-tokens", type=int, default=12)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=0.9)
    args = parser.parse_args()

    modelo, checkpoint, device = cargar_miniai()
    print("Dispositivo:", device)
    print("Checkpoint:", RUTA_CHECKPOINT.name)
    try:
        respuesta = generar(modelo, checkpoint, device, args.prompt,
                            args.max_tokens, args.temperature, args.top_k, args.top_p)
    except ValueError as error:
        parser.error(str(error))
    print("Prompt:", args.prompt)
    print("Generación:", respuesta)


if __name__ == "__main__":
    main()