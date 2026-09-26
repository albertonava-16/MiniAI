import torch
import torch.nn.functional as F


tokens = [
    "el",
    "perro",
    "come"
]


embeddings = torch.tensor([
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0]
])


W_Q = torch.tensor([
    [1.0, 0.0],
    [0.0, 1.0]
])

W_K = torch.tensor([
    [1.0, 0.0],
    [0.0, 1.0]
])

W_V = torch.tensor([
    [1.0, 0.0],
    [0.0, 1.0]
])


Q = embeddings @ W_Q
K = embeddings @ W_K
V = embeddings @ W_V


# --------------------------------------------------
# SCORES DE TODOS CONTRA TODOS
# --------------------------------------------------

scores = Q @ K.T

print("Scores:")
print(scores)


# --------------------------------------------------
# SOFTMAX POR FILA
# --------------------------------------------------

pesos_attention = F.softmax(
    scores,
    dim=1
)

print("\nPesos de attention:")
print(pesos_attention)


# --------------------------------------------------
# COMBINAR VALUES
# --------------------------------------------------

salida = pesos_attention @ V

print("\nSalida self-attention:")
print(salida)


# --------------------------------------------------
# MOSTRAR POR TOKEN
# --------------------------------------------------

print("\nResultado por token:\n")

for token, vector in zip(tokens, salida):
    print(
        f"{token:8} -> {vector}"
    )