import torch


# --------------------------------------------------
# Creamos 3 embeddings manuales
# --------------------------------------------------

tokens = [
    "el",
    "perro",
    "come"
]

embeddings = torch.tensor([
    [1.0, 0.0],   # el
    [0.0, 1.0],   # perro
    [1.0, 1.0]    # come
])


print("Tokens:")
print(tokens)

print("\nEmbeddings:")
print(embeddings)


# --------------------------------------------------
# Elegimos el token "come"
# --------------------------------------------------

query = embeddings[2]

print("\nQuery (come):")
print(query)


# --------------------------------------------------
# Calculamos similitud
# --------------------------------------------------

scores = embeddings @ query

print("\nScores:")
print(scores)


# --------------------------------------------------
# Convertimos scores a pesos
# --------------------------------------------------

pesos_attention = torch.softmax(
    scores,
    dim=0
)

print("\nPesos de attention:")
print(pesos_attention)


# --------------------------------------------------
# Mezclamos información
# --------------------------------------------------

resultado = torch.sum(
    embeddings * pesos_attention.unsqueeze(1),
    dim=0
)

print("\nResultado de attention:")
print(resultado)