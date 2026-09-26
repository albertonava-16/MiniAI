import torch
import torch.nn.functional as F
import math


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
# DIMENSIÓN DE KEY
# --------------------------------------------------

d_k = K.shape[1]


# --------------------------------------------------
# SCORES ESCALADOS
# --------------------------------------------------

scores = (Q @ K.T) / math.sqrt(d_k)

print("Scores escalados:")
print(scores)


# --------------------------------------------------
# SOFTMAX
# --------------------------------------------------

pesos_attention = F.softmax(
    scores,
    dim=1
)

print("\nPesos de attention:")
print(pesos_attention)


# --------------------------------------------------
# RESULTADO
# --------------------------------------------------

salida = pesos_attention @ V

print("\nSalida attention:")
print(salida)


print("\nResultado por token:\n")

for token, vector in zip(tokens, salida):
    print(
        f"{token:8} -> {vector}"
    )