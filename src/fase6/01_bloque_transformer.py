import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# --------------------------------------------------
# UNA CABEZA DE ATTENTION
# --------------------------------------------------

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

        return salida


# --------------------------------------------------
# MULTI-HEAD ATTENTION
# --------------------------------------------------

class MultiHeadAttention(nn.Module):

    def __init__(
        self,
        embedding_dim,
        num_heads
    ):
        super().__init__()

        assert embedding_dim % num_heads == 0

        head_dim = (
            embedding_dim // num_heads
        )

        self.heads = nn.ModuleList([
            AttentionHead(
                embedding_dim,
                head_dim
            )
            for _ in range(num_heads)
        ])

        self.proyeccion = nn.Linear(
            embedding_dim,
            embedding_dim
        )


    def forward(self, x):

        salidas = [
            head(x)
            for head in self.heads
        ]

        combinado = torch.cat(
            salidas,
            dim=1
        )

        return self.proyeccion(
            combinado
        )


# --------------------------------------------------
# FEED FORWARD
# --------------------------------------------------

class FeedForward(nn.Module):

    def __init__(self, embedding_dim):
        super().__init__()

        self.red = nn.Sequential(

            nn.Linear(
                embedding_dim,
                embedding_dim * 4
            ),

            nn.ReLU(),

            nn.Linear(
                embedding_dim * 4,
                embedding_dim
            )
        )


    def forward(self, x):
        return self.red(x)


# --------------------------------------------------
# BLOQUE TRANSFORMER
# --------------------------------------------------

class TransformerBlock(nn.Module):

    def __init__(
        self,
        embedding_dim,
        num_heads
    ):
        super().__init__()

        self.attention = MultiHeadAttention(
            embedding_dim,
            num_heads
        )

        self.feed_forward = FeedForward(
            embedding_dim
        )

        self.norm1 = nn.LayerNorm(
            embedding_dim
        )

        self.norm2 = nn.LayerNorm(
            embedding_dim
        )


    def forward(self, x):

        # Attention
        attention_output = self.attention(x)

        # Residual + LayerNorm
        x = self.norm1(
            x + attention_output
        )

        # Feed Forward
        ff_output = self.feed_forward(x)

        # Residual + LayerNorm
        x = self.norm2(
            x + ff_output
        )

        return x


# --------------------------------------------------
# EJEMPLO
# --------------------------------------------------

tokens = [
    "el",
    "perro",
    "come"
]


embeddings = torch.tensor([
    [1.0, 0.0, 0.5, 0.2],
    [0.0, 1.0, 0.3, 0.7],
    [1.0, 1.0, 0.8, 0.6]
])


modelo = TransformerBlock(
    embedding_dim=4,
    num_heads=2
)


salida = modelo(
    embeddings
)


print("Entrada:")
print(embeddings)

print("\nSalida Transformer:")
print(salida)

print("\nForma:")
print(salida.shape)