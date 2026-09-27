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

        return pesos @ V


class MultiHeadAttention(nn.Module):

    def __init__(self, embedding_dim, num_heads):
        super().__init__()

        assert embedding_dim % num_heads == 0

        head_dim = embedding_dim // num_heads

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


class TransformerBlock(nn.Module):

    def __init__(self, embedding_dim, num_heads):
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

        attention_output = self.attention(x)

        x = self.norm1(
            x + attention_output
        )

        ff_output = self.feed_forward(x)

        x = self.norm2(
            x + ff_output
        )

        return x


class MiniGPT(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        num_heads,
        max_seq_len,
        num_blocks
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(
            vocab_size,
            embedding_dim
        )

        self.position_embedding = nn.Embedding(
            max_seq_len,
            embedding_dim
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(
                embedding_dim,
                num_heads
            )
            for _ in range(num_blocks)
        ])

        # NUEVO:
        # convertir cada vector a logits del vocabulario
        self.salida = nn.Linear(
            embedding_dim,
            vocab_size
        )

    def forward(self, token_ids):

        seq_len = token_ids.shape[0]

        posiciones = torch.arange(
            seq_len,
            device=token_ids.device
        )

        token_vectors = self.token_embedding(
            token_ids
        )

        position_vectors = self.position_embedding(
            posiciones
        )

        x = token_vectors + position_vectors

        for block in self.blocks:
            x = block(x)

        logits = self.salida(x)

        return logits


# --------------------------------------------------
# VOCABULARIO
# --------------------------------------------------

vocabulario = [
    "el",
    "perro",
    "come"
]

token_a_id = {
    token: indice
    for indice, token in enumerate(vocabulario)
}

id_a_token = {
    indice: token
    for token, indice in token_a_id.items()
}


# --------------------------------------------------
# ENTRADA
# --------------------------------------------------

token_ids = torch.tensor([
    token_a_id["el"],
    token_a_id["perro"],
    token_a_id["come"]
])


# --------------------------------------------------
# MODELO
# --------------------------------------------------

modelo = MiniGPT(
    vocab_size=len(vocabulario),
    embedding_dim=4,
    num_heads=2,
    max_seq_len=10,
    num_blocks=3
)


logits = modelo(
    token_ids
)


print("Logits:")
print(logits)

print("\nForma:")
print(logits.shape)


# --------------------------------------------------
# PROBABILIDADES
# --------------------------------------------------

probabilidades = F.softmax(
    logits,
    dim=1
)

print("\nProbabilidades:")
print(probabilidades)


# --------------------------------------------------
# PREDICCIÓN
# --------------------------------------------------

predicciones = torch.argmax(
    probabilidades,
    dim=1
)


print("\nPredicciones:")

for token_actual, pred_id in zip(
    ["el", "perro", "come"],
    predicciones
):

    print(
        token_actual,
        "->",
        id_a_token[pred_id.item()]
    )