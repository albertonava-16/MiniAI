import torch
import torch.nn as nn
import torch.nn.functional as F


vocabulario = [
    "hola",
    "ia",
    "mundo"
]

token_a_id = {
    token: indice
    for indice, token in enumerate(vocabulario)
}

embedding = nn.Embedding(
    num_embeddings=len(vocabulario),
    embedding_dim=4
)


id_hola = torch.tensor(token_a_id["hola"])
id_ia = torch.tensor(token_a_id["ia"])
id_mundo = torch.tensor(token_a_id["mundo"])


vector_hola = embedding(id_hola)
vector_ia = embedding(id_ia)
vector_mundo = embedding(id_mundo)


print("Embedding hola:")
print(vector_hola)

print("\nEmbedding ia:")
print(vector_ia)

print("\nEmbedding mundo:")
print(vector_mundo)


sim_hola_ia = F.cosine_similarity(
    vector_hola.unsqueeze(0),
    vector_ia.unsqueeze(0)
)

sim_hola_mundo = F.cosine_similarity(
    vector_hola.unsqueeze(0),
    vector_mundo.unsqueeze(0)
)


print("\nSimilitud hola - ia:")
print(sim_hola_ia.item())

print("\nSimilitud hola - mundo:")
print(sim_hola_mundo.item())