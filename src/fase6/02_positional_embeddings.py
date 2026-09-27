import torch
import torch.nn as nn


# --------------------------------------------------
# TOKENS DE EJEMPLO
# --------------------------------------------------

tokens = [
    "el",
    "perro",
    "come"
]


# --------------------------------------------------
# EMBEDDINGS DE TOKENS
# --------------------------------------------------

token_embeddings = torch.tensor([
    [1.0, 0.0, 0.5, 0.2],
    [0.0, 1.0, 0.3, 0.7],
    [1.0, 1.0, 0.8, 0.6]
])


embedding_dim = 4
seq_len = len(tokens)


# --------------------------------------------------
# EMBEDDINGS DE POSICIÓN
# --------------------------------------------------

position_embedding = nn.Embedding(
    num_embeddings=seq_len,
    embedding_dim=embedding_dim
)


# posiciones:
# [0, 1, 2]

posiciones = torch.arange(seq_len)


print("Posiciones:")
print(posiciones)


# --------------------------------------------------
# CONVERTIR POSICIONES A VECTORES
# --------------------------------------------------

position_vectors = position_embedding(
    posiciones
)


print("\nEmbeddings de posición:")
print(position_vectors)


# --------------------------------------------------
# SUMAR TOKEN + POSICIÓN
# --------------------------------------------------

entrada_final = (
    token_embeddings
    +
    position_vectors
)


print("\nToken embeddings:")
print(token_embeddings)


print("\nToken + posición:")
print(entrada_final)


print("\nForma final:")
print(entrada_final.shape)