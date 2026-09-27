import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# --------------------------------------------------
# ATTENTION HEAD
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

        salida = self.proyeccion(
            combinado
        )

        return salida


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
# TRANSFORMER BLOCK
# --------------------------------------------------

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


# --------------------------------------------------
# MINI GPT
# --------------------------------------------------

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

        x = (
            token_vectors
            +
            position_vectors
        )

        for block in self.blocks:
            x = block(x)

        logits = self.salida(x)

        return logits


# --------------------------------------------------
# DISPOSITIVO
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Usando:", device)


# --------------------------------------------------
# DATOS
# --------------------------------------------------

texto = """
el perro come el gato duerme
"""

tokens = texto.split()


# --------------------------------------------------
# VOCABULARIO
# --------------------------------------------------

vocabulario = sorted(
    set(tokens)
)

token_a_id = {
    token: indice
    for indice, token
    in enumerate(vocabulario)
}

id_a_token = {
    indice: token
    for token, indice
    in token_a_id.items()
}


print("\nVocabulario:")
print(vocabulario)


# --------------------------------------------------
# CONVERTIR TEXTO A IDS
# --------------------------------------------------

ids = [
    token_a_id[token]
    for token in tokens
]


# --------------------------------------------------
# ENTRADA Y OBJETIVO
# --------------------------------------------------

# entrada:
# el perro come el gato

X = torch.tensor(
    ids[:-1],
    dtype=torch.long
).to(device)


# objetivo:
# perro come el gato duerme

y = torch.tensor(
    ids[1:],
    dtype=torch.long
).to(device)


print("\nEntrada:")

for token_id in X:

    print(
        id_a_token[token_id.item()],
        end=" "
    )


print("\n\nObjetivo:")

for token_id in y:

    print(
        id_a_token[token_id.item()],
        end=" "
    )


print()


# --------------------------------------------------
# CREAR MODELO
# --------------------------------------------------

modelo = MiniGPT(
    vocab_size=len(vocabulario),
    embedding_dim=8,
    num_heads=2,
    max_seq_len=20,
    num_blocks=2
).to(device)


# --------------------------------------------------
# LOSS
# --------------------------------------------------

criterio = nn.CrossEntropyLoss()


# --------------------------------------------------
# OPTIMIZER
# --------------------------------------------------

optimizer = torch.optim.Adam(
    modelo.parameters(),
    lr=0.01
)


# --------------------------------------------------
# ENTRENAMIENTO
# --------------------------------------------------

epochs = 2000


for epoch in range(epochs):

    # Forward
    logits = modelo(X)

    # Calcular error
    loss = criterio(
        logits,
        y
    )

    # Limpiar gradientes anteriores
    optimizer.zero_grad()

    # Backpropagation
    loss.backward()

    # Actualizar parámetros
    optimizer.step()


    if epoch % 200 == 0:

        print(
            f"\nEpoch {epoch} "
            f"Loss: {loss.item():.6f}"
        )


# --------------------------------------------------
# LOSS FINAL
# --------------------------------------------------

print(
    "\nLoss final:",
    loss.item()
)


# --------------------------------------------------
# PREDICCIONES
# --------------------------------------------------

modelo.eval()

with torch.no_grad():

    logits = modelo(X)

    probabilidades = F.softmax(
        logits,
        dim=1
    )

    predicciones = torch.argmax(
        probabilidades,
        dim=1
    )


print("\nPredicciones:\n")


for entrada_id, pred_id, objetivo_id in zip(
    X,
    predicciones,
    y
):

    entrada = id_a_token[
        entrada_id.item()
    ]

    prediccion = id_a_token[
        pred_id.item()
    ]

    objetivo = id_a_token[
        objetivo_id.item()
    ]

    print(
        f"{entrada:8} "
        f"-> {prediccion:8} "
        f"(esperado: {objetivo})"
    )