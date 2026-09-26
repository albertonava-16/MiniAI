import torch
import torch.nn.functional as F


# --------------------------------------------------
# TOKENS
# --------------------------------------------------

tokens = [
    "el",
    "perro",
    "come"
]


# --------------------------------------------------
# EMBEDDINGS DE EJEMPLO
# --------------------------------------------------

embeddings = torch.tensor([
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0]
])


print("Embeddings:")
print(embeddings)


# --------------------------------------------------
# MATRICES Q, K, V
# --------------------------------------------------

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


# --------------------------------------------------
# CALCULAR QUERY, KEY Y VALUE
# --------------------------------------------------

Q = embeddings @ W_Q
K = embeddings @ W_K
V = embeddings @ W_V


print("\nQ:")
print(Q)

print("\nK:")
print(K)

print("\nV:")
print(V)


# --------------------------------------------------
# USAMOS EL TOKEN "come" COMO QUERY
# --------------------------------------------------

query_come = Q[2]


# --------------------------------------------------
# COMPARAR QUERY CONTRA TODOS LOS KEYS
# --------------------------------------------------

scores = K @ query_come

print("\nScores para 'come':")
print(scores)


# --------------------------------------------------
# SOFTMAX
# --------------------------------------------------

pesos = F.softmax(
    scores,
    dim=0
)

print("\nPesos de attention:")
print(pesos)


# --------------------------------------------------
# COMBINAR LOS VALUES
# --------------------------------------------------

resultado = torch.sum(
    V * pesos.unsqueeze(1),
    dim=0
)

print("\nResultado attention:")
print(resultado)