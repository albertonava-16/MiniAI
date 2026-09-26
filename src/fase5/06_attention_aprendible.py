import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CausalSelfAttention(nn.Module):

    def __init__(self, embedding_dim):
        super().__init__()

        self.embedding_dim = embedding_dim

        self.W_Q = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=False
        )

        self.W_K = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=False
        )

        self.W_V = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=False
        )


    def forward(self, x):

        # x:
        # [seq_len, embedding_dim]

        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        scores = (
            Q @ K.T
        ) / math.sqrt(self.embedding_dim)

        seq_len = x.shape[0]

        mask = torch.triu(
            torch.ones(
                seq_len,
                seq_len,
                device=x.device
            ),
            diagonal=1
        ).bool()

        scores = scores.masked_fill(
            mask,
            float("-inf")
        )

        pesos = F.softmax(
            scores,
            dim=1
        )

        salida = pesos @ V

        return salida, pesos


# --------------------------------------------------
# DATOS DE EJEMPLO
# --------------------------------------------------

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


# --------------------------------------------------
# MODELO
# --------------------------------------------------

attention = CausalSelfAttention(
    embedding_dim=2
)


salida, pesos = attention(
    embeddings
)


print("Pesos de attention:")
print(pesos)

print("\nSalida:")
print(salida)


print("\nResultado por token:\n")

for token, vector in zip(tokens, salida):
    print(
        f"{token:8} -> {vector}"
    )