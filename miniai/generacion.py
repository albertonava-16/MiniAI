
import torch


def seleccionar_token(
    logits,
    temperature=1.0,
    top_k=None,
    top_p=0.9
):
    """
    Selecciona el siguiente token aplicando
    temperature, top-k y top-p.
    """

    if temperature <= 0:
        raise ValueError("temperature debe ser mayor que 0")

    if top_k is not None and (
        not isinstance(top_k, int) or top_k < 1
    ):
        raise ValueError("top_k debe ser un entero positivo")

    if not (0 < top_p <= 1):
        raise ValueError("top_p debe estar entre 0 y 1")

    logits = logits / temperature

    # Aplicar top-k
    if top_k is not None:
        k = min(top_k, logits.numel())

        minimo = torch.topk(logits, k).values[-1]

        logits = logits.masked_fill(
            logits < minimo,
            float("-inf")
        )

    # Convertir logits en probabilidades
    probabilidades = torch.softmax(logits, dim=-1)

    # Ordenar de mayor a menor probabilidad
    probs_ordenadas, indices_ordenados = torch.sort(
        probabilidades,
        descending=True
    )

    acumuladas = torch.cumsum(
        probs_ordenadas,
        dim=-1
    )

    # Top-p corregido: conservar el token
    # que permite cruzar el umbral
    eliminar = acumuladas > top_p
    eliminar[1:] = eliminar[:-1].clone()
    eliminar[0] = False

    probs_ordenadas = probs_ordenadas.masked_fill(
        eliminar,
        0
    )

    probs_ordenadas = (
        probs_ordenadas / probs_ordenadas.sum()
    )

    indice = torch.multinomial(
        probs_ordenadas,
        num_samples=1
    ).item()

    return indices_ordenados[indice].item()


def generar(
    modelo,
    checkpoint,
    device,
    prompt,
    max_tokens=12,
    temperature=1.0,
    top_k=None,
    top_p=0.9
):
    """
    Genera una continuación de texto utilizando
    un MiniGPT previamente cargado.
    """

    if not isinstance(max_tokens, int) or max_tokens < 1:
        raise ValueError("max_tokens debe ser positivo")

    vocabulario = checkpoint["vocabulario"]

    token_a_id = {
        token: i
        for i, token in enumerate(vocabulario)
    }

    especiales = checkpoint["special_tokens"]

    BOS = especiales["BOS"]
    EOS = especiales["EOS"]

    tokens_prompt = prompt.strip().lower().split()

    if not tokens_prompt:
        raise ValueError("El prompt no puede estar vacío")

    desconocidos = [
        token
        for token in tokens_prompt
        if token not in token_a_id
    ]

    if desconocidos:
        raise ValueError(
            f"Tokens desconocidos: {desconocidos}. "
            "Usa palabras del corpus de Fase 7."
        )

    if BOS in tokens_prompt or EOS in tokens_prompt:
        raise ValueError(
            "No introduzcas BOS ni EOS manualmente"
        )

    ids = (
        [token_a_id[BOS]]
        + [token_a_id[token] for token in tokens_prompt]
    )

    max_contexto = checkpoint["config"]["max_seq_len"]

    generados = []

    with torch.inference_mode():

        for _ in range(max_tokens):

            # Respetar la longitud máxima de contexto
            contexto = ids[-max_contexto:]

            # Completar con BOS cuando falten posiciones
            contexto = (
                [token_a_id[BOS]]
                * (max_contexto - len(contexto))
                + contexto
            )

            entrada = torch.tensor(
                [contexto],
                dtype=torch.long,
                device=device
            )

            logits = modelo(entrada)[0, -1, :].clone()

            # BOS no debe generarse como continuación
            logits[token_a_id[BOS]] = float("-inf")

            siguiente = seleccionar_token(
                logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )

            if siguiente == token_a_id[EOS]:
                break

            ids.append(siguiente)

            generados.append(vocabulario[siguiente])

    return " ".join(tokens_prompt + generados)
