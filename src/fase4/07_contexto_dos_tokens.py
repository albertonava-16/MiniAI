import torch
import torch.nn as nn


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Usando:", device)


texto = """
el perro come
el gato duerme
el perro duerme
el gato come
"""

tokens = texto.split()

vocabulario = sorted(set(tokens))

token_a_id = {
    token: indice
    for indice, token in enumerate(vocabulario)
}

id_a_token = {
    indice: token
    for token, indice in token_a_id.items()
}


# --------------------------------------------------
# CONTEXTO DE 2 TOKENS
# --------------------------------------------------

context_size = 2

entradas = []
objetivos = []

for i in range(len(tokens) - context_size):

    contexto = tokens[i:i + context_size]

    objetivo = tokens[i + context_size]

    entradas.append([
        token_a_id[token]
        for token in contexto
    ])

    objetivos.append(
        token_a_id[objetivo]
    )


X = torch.tensor(
    entradas,
    dtype=torch.long
).to(device)

y = torch.tensor(
    objetivos,
    dtype=torch.long
).to(device)


print("\nEjemplos de entrenamiento:\n")

for contexto_ids, objetivo_id in zip(X, y):

    contexto_tokens = [
        id_a_token[id.item()]
        for id in contexto_ids
    ]

    print(
        contexto_tokens,
        "->",
        id_a_token[objetivo_id.item()]
    )


class MiniModeloContexto(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        context_size
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim
        )

        self.salida = nn.Linear(
            embedding_dim * context_size,
            vocab_size
        )

    def forward(self, x):

        x = self.embedding(x)

        # x tiene forma:
        # batch, context_size, embedding_dim

        x = x.flatten(start_dim=1)

        x = self.salida(x)

        return x


modelo = MiniModeloContexto(
    vocab_size=len(vocabulario),
    embedding_dim=8,
    context_size=context_size
).to(device)


criterio = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    modelo.parameters(),
    lr=0.05
)


epochs = 3000

for epoch in range(epochs):

    logits = modelo(X)

    loss = criterio(logits, y)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()


print("\nLoss final:")
print(loss.item())


# --------------------------------------------------
# FUNCIÓN PARA PROBAR CONTEXTO
# --------------------------------------------------

def predecir(token1, token2):

    entrada = torch.tensor(
        [[
            token_a_id[token1],
            token_a_id[token2]
        ]],
        dtype=torch.long
    ).to(device)

    with torch.no_grad():

        logits = modelo(entrada)

        probabilidades = torch.softmax(
            logits,
            dim=1
        )

        prediccion_id = torch.argmax(
            probabilidades,
            dim=1
        ).item()

    return id_a_token[prediccion_id]


print("\nPredicciones:\n")

print(
    "el perro ->",
    predecir("el", "perro")
)

print(
    "el gato ->",
    predecir("el", "gato")
)