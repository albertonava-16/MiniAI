import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# --------------------------------------------------
# DEVICE
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Usando:", device)


# --------------------------------------------------
# CORPUS
# --------------------------------------------------

texto = """
el perro corre por el parque
el gato duerme en la casa
el perro come su comida
el gato mira por la ventana
el perro juega con la pelota
el gato come pescado
el perro duerme en el suelo
el gato corre por la casa
"""


tokens = texto.split()

vocabulario = sorted(set(tokens))

token_a_id = {
    token: indice
    for indice, token in enumerate(vocabulario)
}

id_a_token = {
    indice: token
    for token, indice in token_a_id.items()
}


ids = [
    token_a_id[token]
    for token in tokens
]

datos = torch.tensor(
    ids,
    dtype=torch.long
)


# --------------------------------------------------
# TRAIN / VALIDATION
# --------------------------------------------------

division = int(
    len(datos) * 0.8
)

train_data = datos[:division]

validation_data = datos[division:]


# --------------------------------------------------
# CREAR EJEMPLOS
# --------------------------------------------------

context_size = 4


def crear_ejemplos(data, context_size):

    entradas = []
    objetivos = []

    for i in range(
        len(data) - context_size
    ):

        entrada = data[
            i:i + context_size
        ]

        objetivo = data[
            i + 1:
            i + context_size + 1
        ]

        entradas.append(entrada)
        objetivos.append(objetivo)

    X = torch.stack(entradas)
    y = torch.stack(objetivos)

    return X, y


X_train, y_train = crear_ejemplos(
    train_data,
    context_size
)

X_val, y_val = crear_ejemplos(
    validation_data,
    context_size
)


X_train = X_train.to(device)
y_train = y_train.to(device)

X_val = X_val.to(device)
y_val = y_val.to(device)


print("\nX_train:")
print(X_train.shape)

print("X_val:")
print(X_val.shape)


# --------------------------------------------------
# ATTENTION HEAD
# --------------------------------------------------

class AttentionHead(nn.Module):

    def __init__(
        self,
        embedding_dim,
        head_dim
    ):
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

        # x:
        # [batch, seq_len, embedding_dim]

        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        # [batch, seq, seq]

        scores = (
            Q @ K.transpose(-2, -1)
        ) / math.sqrt(self.head_dim)

        seq_len = x.shape[1]

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
            dim=-1
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

        assert (
            embedding_dim % num_heads == 0
        )

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
            dim=-1
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
# TRANSFORMER BLOCK
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

        # token_ids:
        # [batch, seq_len]

        seq_len = token_ids.shape[1]

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

        # position_vectors:
        # [seq_len, embedding_dim]
        #
        # PyTorch lo suma automáticamente
        # a cada elemento del batch.

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
# MODELO
# --------------------------------------------------

modelo = MiniGPT(
    vocab_size=len(vocabulario),
    embedding_dim=16,
    num_heads=4,
    max_seq_len=context_size,
    num_blocks=2
).to(device)


criterio = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    modelo.parameters(),
    lr=0.01
)


# --------------------------------------------------
# FUNCIÓN PARA CALCULAR LOSS
# --------------------------------------------------

def calcular_loss(X, y):

    logits = modelo(X)

    # logits:
    # [batch, seq_len, vocab_size]

    # CrossEntropy quiere:
    # [ejemplos, clases]

    logits = logits.reshape(
        -1,
        len(vocabulario)
    )

    targets = y.reshape(-1)

    loss = criterio(
        logits,
        targets
    )

    return loss


# --------------------------------------------------
# ENTRENAMIENTO
# --------------------------------------------------

epochs = 1000


print("\nEntrenando...\n")


for epoch in range(epochs):

    modelo.train()

    train_loss = calcular_loss(
        X_train,
        y_train
    )

    optimizer.zero_grad()

    train_loss.backward()

    optimizer.step()


    # ----------------------------------------------
    # VALIDACIÓN
    # ----------------------------------------------

    if epoch % 100 == 0:

        modelo.eval()

        with torch.no_grad():

            validation_loss = calcular_loss(
                X_val,
                y_val
            )

        print(
            f"Epoch {epoch:4d} | "
            f"Train Loss: {train_loss.item():.4f} | "
            f"Validation Loss: {validation_loss.item():.4f}"
        )


# --------------------------------------------------
# RESULTADO FINAL
# --------------------------------------------------

modelo.eval()

with torch.no_grad():

    train_loss = calcular_loss(
        X_train,
        y_train
    )

    validation_loss = calcular_loss(
        X_val,
        y_val
    )


print("\nResultado final:")

print(
    "Train Loss:",
    train_loss.item()
)

print(
    "Validation Loss:",
    validation_loss.item()
)