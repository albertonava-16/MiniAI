import torch
import torch.nn as nn


# --------------------------------------------------
# VOCABULARIO
# --------------------------------------------------

vocabulario = [
    "hola",
    "ia",
    "mundo"
]

token_a_id = {
    token: indice
    for indice, token in enumerate(vocabulario)
}

print("Token -> ID:")
print(token_a_id)


# --------------------------------------------------
# IDS DE EJEMPLO
# --------------------------------------------------

ids = torch.tensor([
    token_a_id["hola"],
    token_a_id["mundo"],
    token_a_id["ia"]
])

print("\nIDs:")
print(ids)


# --------------------------------------------------
# CAPA DE EMBEDDING
# --------------------------------------------------

embedding = nn.Embedding(
    num_embeddings=len(vocabulario),
    embedding_dim=4
)


# --------------------------------------------------
# CONVERTIR IDS A VECTORES
# --------------------------------------------------

vectores = embedding(ids)

print("\nEmbeddings:")
print(vectores)