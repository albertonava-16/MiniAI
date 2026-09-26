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


d_k = K.shape[1]


# --------------------------------------------------
# SCORES
# --------------------------------------------------

scores = (Q @ K.T) / math.sqrt(d_k)

print("Scores originales:")
print(scores)


# --------------------------------------------------
# MÁSCARA CAUSAL
# --------------------------------------------------

mask = torch.triu(
    torch.ones_like(scores),
    diagonal=1
).bool()

print("\nMáscara causal:")
print(mask)


# --------------------------------------------------
# BLOQUEAR FUTURO
# --------------------------------------------------

scores_mascarados = scores.masked_fill(
    mask,
    float("-inf")
)

print("\nScores con máscara:")
print(scores_mascarados)


# --------------------------------------------------
# SOFTMAX
# --------------------------------------------------

pesos_attention = F.softmax(
    scores_mascarados,
    dim=1
)

print("\nPesos de attention:")
print(pesos_attention)


# --------------------------------------------------
# RESULTADO
# --------------------------------------------------

salida = pesos_attention @ V

print("\nSalida causal attention:")
print(salida)


print("\nResultado por token:\n")

for token, vector in zip(tokens, salida):
    print(
        f"{token:8} -> {vector}"
    )