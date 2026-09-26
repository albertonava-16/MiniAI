import torch
import torch.nn as nn


# --------------------------------------------------
# DISPOSITIVO
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Usando:", device)


# --------------------------------------------------
# CORPUS
# --------------------------------------------------

texto = """
el perro come
el gato come
el perro duerme
el gato duerme
"""

tokens = texto.split()


# --------------------------------------------------
# VOCABULARIO
# --------------------------------------------------

vocabulario = sorted(set(tokens))

token_a_id = {
    token: indice
    for indice, token in enumerate(vocabulario)
}

id_a_token = {
    indice: token
    for token, indice in token_a_id.items()
}

print("\nVocabulario:")
print(vocabulario)


# --------------------------------------------------
# PARES DE ENTRENAMIENTO
# token actual -> siguiente token
# --------------------------------------------------

entradas = []
objetivos = []

for i in range(len(tokens) - 1):

    entrada = token_a_id[tokens[i]]
    objetivo = token_a_id[tokens[i + 1]]

    entradas.append(entrada)
    objetivos.append(objetivo)


X = torch.tensor(
    entradas,
    dtype=torch.long
).to(device)

y = torch.tensor(
    objetivos,
    dtype=torch.long
).to(device)


# --------------------------------------------------
# MODELO
# --------------------------------------------------

class MiniModeloLenguaje(nn.Module):

    def __init__(self, vocab_size, embedding_dim):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim
        )

        self.salida = nn.Linear(
            embedding_dim,
            vocab_size
        )

    def forward(self, x):

        x = self.embedding(x)

        x = self.salida(x)

        return x


modelo = MiniModeloLenguaje(
    vocab_size=len(vocabulario),
    embedding_dim=8
).to(device)


# --------------------------------------------------
# LOSS Y OPTIMIZER
# --------------------------------------------------

criterio = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    modelo.parameters(),
    lr=0.05
)


# --------------------------------------------------
# ENTRENAMIENTO
# --------------------------------------------------

epochs = 3000

for epoch in range(epochs):

    logits = modelo(X)

    loss = criterio(logits, y)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()


# --------------------------------------------------
# RESULTADO
# --------------------------------------------------

print("\nLoss final:")
print(loss.item())


# --------------------------------------------------
# PROBAR PREDICCIONES
# --------------------------------------------------

print("\nPredicciones:\n")

for token in vocabulario:

    token_id = torch.tensor(
        [token_a_id[token]],
        dtype=torch.long
    ).to(device)

    with torch.no_grad():

        logits = modelo(token_id)

        probabilidades = torch.softmax(
            logits,
            dim=1
        )

        prediccion_id = torch.argmax(
            probabilidades,
            dim=1
        ).item()

    print(
        f"{token:8} -> "
        f"{id_a_token[prediccion_id]}"
    )