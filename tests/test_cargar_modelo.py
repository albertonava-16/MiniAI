import torch

from miniai.inferencia import cargar_modelo


def main():

    modelo, checkpoint, device = cargar_modelo()

    vocabulario = checkpoint["vocabulario"]

    BOS = checkpoint["special_tokens"]["BOS"]

    token_a_id = {
        token: i
        for i, token in enumerate(vocabulario)
    }

    entrada = torch.tensor(
        [[token_a_id[BOS]]],
        dtype=torch.long,
        device=device
    )

    with torch.inference_mode():
        logits = modelo(entrada)

    assert logits.shape == (1, 1, len(vocabulario))
    assert torch.isfinite(logits).all().item()

    print("Dispositivo:", device)
    print("Vocabulario:", len(vocabulario))
    print("Logits:", tuple(logits.shape))
    print("Modelo cargado correctamente.")
    print("PRUEBA COMPLETADA")


if __name__ == "__main__":
    main()