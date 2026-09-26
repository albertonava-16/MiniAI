import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class AttentionHead(nn.Module):

    def __init__(self, embedding_dim, head_dim):
        super().__init__()

        self.head_dim = head_dim

        self.W_Q = nn.Linear(
            embedding_dim,
            head_dim,
            bias=False
        )

        self.W_K = nn.Linear(
            embedding_dim,
            head_dim,
            bias=False
        )

        self.W_V = nn.Linear(
            embedding_dim,
            head_dim,
            bias=False
        )


    def forward(self, x):

        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        scores = (
            Q @ K.T
        ) / math.sqrt(self.head_dim)

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


class MultiHeadAttention(nn.Module):

    def __init__(
        self,
        embedding_dim,
        num_heads
    ):
        super().__init__()

        assert embedding_dim % num_heads == 0

        self.head_dim = (
            embedding_dim // num_heads
        )

        self.heads = nn.ModuleList([
            AttentionHead(
                embedding_dim,
                self.head_dim
            )
            for _ in range(num_heads)
        ])


        self.proyeccion = nn.Linear(
            embedding_dim,
            embedding_dim
        )


    def forward(self, x):

        salidas = []
        pesos_heads = []

        for head in self.heads:

            salida, pesos = head(x)

            salidas.append(salida)
            pesos_heads.append(pesos)


        # Juntar las cabezas
        combinado = torch.cat(
            salidas,
            dim=1
        )


        salida_final = self.proyeccion(
            combinado
        )

        return salida_final, pesos_heads


# --------------------------------------------------
# EJEMPLO
# --------------------------------------------------

tokens = [
    "el",
    "perro",
    "come"
]


# Ahora usamos embeddings de dimensión 4
embeddings = torch.tensor([
    [1.0, 0.0, 0.5, 0.2],
    [0.0, 1.0, 0.3, 0.7],
    [1.0, 1.0, 0.8, 0.6]
])


modelo = MultiHeadAttention(
    embedding_dim=4,
    num_heads=2
)


salida, pesos_heads = modelo(
    embeddings
)


print("Salida Multi-Head Attention:")
print(salida)


print("\nForma de salida:")
print(salida.shape)


for i, pesos in enumerate(
    pesos_heads
):

    print(
        f"\nPesos Head {i + 1}:"
    )

    print(pesos)