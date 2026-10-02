import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import TensorDataset, DataLoader

import math
import copy


# --------------------------------------------------
# DEVICE
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Usando:", device)


# --------------------------------------------------
# DATASET COMPLETO
# --------------------------------------------------

oraciones = [

    "el perro corre por el parque",
    "el perro juega con la pelota",
    "el perro duerme en la casa",
    "el perro come su comida",
    "el perro mira por la ventana",

    "el gato corre por la casa",
    "el gato juega con la pelota",
    "el gato duerme en el suelo",
    "el gato come pescado",
    "el gato mira por la ventana",

    "la niña juega en el parque",
    "la niña corre por la casa",
    "la niña come en la cocina",
    "la niña mira por la ventana",
    "la niña duerme en la casa",

    "el niño juega en el parque",
    "el niño corre con el perro",
    "el niño come en la cocina",
    "el niño mira al gato",
    "el niño duerme en la casa",

    "el perro juega con el niño",
    "el gato juega con la niña",
    "la niña corre con el perro",
    "el niño juega con el gato",

    "el perro come en la cocina",
    "el gato duerme en la casa",
    "la niña juega con el gato",
    "el niño corre por el parque",

    "el perro mira al gato",
    "el gato mira al perro",
]


# --------------------------------------------------
# VALIDATION
# --------------------------------------------------
#
# Elegimos frases completas.
#
# Las palabras utilizadas también aparecen en otras
# frases de entrenamiento.
#
# Así podemos medir:
#
# ¿aprendió patrones?
#
# y no simplemente:
#
# ¿conoce una palabra nueva?
# --------------------------------------------------

val_oraciones = [

    "el perro corre por el parque",
    "el gato duerme en la casa",
    "la niña juega en el parque",
    "el niño corre con el perro",
    "el perro mira al gato",
    "el gato mira al perro",
]


train_oraciones = [
    frase
    for frase in oraciones
    if frase not in val_oraciones
]


print("\nOraciones Train:")
print(len(train_oraciones))

print("\nOraciones Validation:")
print(len(val_oraciones))


# --------------------------------------------------
# VOCABULARIO
# --------------------------------------------------

tokens_totales = []

for frase in oraciones:

    tokens_totales.extend(
        frase.split()
    )


vocabulario = sorted(
    set(tokens_totales)
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


print("\nTamaño vocabulario:")
print(len(vocabulario))


# --------------------------------------------------
# CONFIGURACIÓN DEL DATASET
# --------------------------------------------------

context_size = 4


# --------------------------------------------------
# CREAR EJEMPLOS POR ORACIÓN
# --------------------------------------------------
#
# IMPORTANTE:
#
# No concatenamos las frases.
#
# Por ejemplo:
#
# "el perro corre por el parque"
#
# genera:
#
# X:
# el perro corre por
#
# Y:
# perro corre por el
#
# después:
#
# X:
# perro corre por el
#
# Y:
# corre por el parque
#
# --------------------------------------------------

def crear_ejemplos(oraciones):

    entradas = []
    objetivos = []

    for frase in oraciones:

        tokens = frase.split()

        ids = [
            token_a_id[token]
            for token in tokens
        ]

        if len(ids) <= context_size:
            continue

        for i in range(
            len(ids) - context_size
        ):

            entrada = ids[
                i:i + context_size
            ]

            objetivo = ids[
                i + 1:
                i + context_size + 1
            ]

            entradas.append(
                entrada
            )

            objetivos.append(
                objetivo
            )


    X = torch.tensor(
        entradas,
        dtype=torch.long
    )

    y = torch.tensor(
        objetivos,
        dtype=torch.long
    )

    return X, y


X_train, y_train = crear_ejemplos(
    train_oraciones
)

X_val, y_val = crear_ejemplos(
    val_oraciones
)


print("\nX_train:")
print(X_train.shape)

print("y_train:")
print(y_train.shape)

print("\nX_val:")
print(X_val.shape)

print("y_val:")
print(y_val.shape)


# --------------------------------------------------
# MINI-BATCHES
# --------------------------------------------------

batch_size = 8


train_dataset = TensorDataset(
    X_train,
    y_train
)


val_dataset = TensorDataset(
    X_val,
    y_val
)


train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)


val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False
)


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
        #
        # [batch, seq_len, embedding_dim]

        Q = self.W_Q(x)

        K = self.W_K(x)

        V = self.W_V(x)


        # --------------------------------------------------
        # SCALED DOT-PRODUCT ATTENTION
        # --------------------------------------------------

        scores = (
            Q @ K.transpose(-2, -1)
        ) / math.sqrt(
            self.head_dim
        )


        seq_len = x.shape[1]


        # --------------------------------------------------
        # CAUSAL MASK
        # --------------------------------------------------

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
        num_heads,
        dropout
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


        self.dropout = nn.Dropout(
            dropout
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


        salida = self.proyeccion(
            combinado
        )


        salida = self.dropout(
            salida
        )


        return salida


# --------------------------------------------------
# FEED FORWARD
# --------------------------------------------------

class FeedForward(nn.Module):

    def __init__(
        self,
        embedding_dim,
        dropout
    ):

        super().__init__()


        self.red = nn.Sequential(

            nn.Linear(
                embedding_dim,
                embedding_dim * 4
            ),

            nn.ReLU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                embedding_dim * 4,
                embedding_dim
            ),

            nn.Dropout(
                dropout
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
        num_heads,
        dropout
    ):

        super().__init__()


        self.attention = MultiHeadAttention(
            embedding_dim,
            num_heads,
            dropout
        )


        self.feed_forward = FeedForward(
            embedding_dim,
            dropout
        )


        self.norm1 = nn.LayerNorm(
            embedding_dim
        )


        self.norm2 = nn.LayerNorm(
            embedding_dim
        )


    def forward(self, x):

        attention_output = self.attention(
            x
        )


        # Residual + LayerNorm

        x = self.norm1(
            x + attention_output
        )


        ff_output = self.feed_forward(
            x
        )


        # Residual + LayerNorm

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
        num_blocks,
        dropout
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
                num_heads,
                dropout
            )

            for _ in range(
                num_blocks
            )

        ])


        self.salida = nn.Linear(
            embedding_dim,
            vocab_size
        )


    def forward(self, token_ids):

        # token_ids:
        #
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


        x = (
            token_vectors
            +
            position_vectors
        )


        for block in self.blocks:

            x = block(x)


        logits = self.salida(
            x
        )


        return logits


# --------------------------------------------------
# MODELO
# --------------------------------------------------

modelo = MiniGPT(

    vocab_size=len(vocabulario),

    embedding_dim=16,

    num_heads=4,

    max_seq_len=context_size,

    num_blocks=2,

    dropout=0.20

).to(device)


# --------------------------------------------------
# LOSS
# --------------------------------------------------

criterio = nn.CrossEntropyLoss()


# --------------------------------------------------
# OPTIMIZER
# --------------------------------------------------
#
# AdamW:
#
# Adam
# +
# weight decay
#
# --------------------------------------------------

optimizer = torch.optim.AdamW(

    modelo.parameters(),

    lr=0.003,

    weight_decay=0.01
)


# --------------------------------------------------
# CALCULAR LOSS DE UN BATCH
# --------------------------------------------------

def calcular_loss_batch(
    X,
    y
):

    logits = modelo(X)


    # logits:
    #
    # [batch, seq_len, vocab_size]


    logits = logits.reshape(
        -1,
        len(vocabulario)
    )


    targets = y.reshape(
        -1
    )


    loss = criterio(
        logits,
        targets
    )


    return loss


# --------------------------------------------------
# EVALUAR DATASET COMPLETO
# --------------------------------------------------

def evaluar(loader):

    modelo.eval()


    perdida_total = 0

    cantidad_batches = 0


    with torch.no_grad():

        for X, y in loader:

            X = X.to(device)

            y = y.to(device)


            loss = calcular_loss_batch(
                X,
                y
            )


            perdida_total += (
                loss.item()
            )


            cantidad_batches += 1


    return (
        perdida_total
        /
        cantidad_batches
    )


# --------------------------------------------------
# ENTRENAMIENTO
# --------------------------------------------------

epochs = 500


# --------------------------------------------------
# EARLY STOPPING
# --------------------------------------------------

mejor_validation_loss = float(
    "inf"
)


mejor_modelo = None


patience = 8


sin_mejorar = 0


evaluar_cada = 20


print("\nEntrenando...\n")


for epoch in range(epochs):

    modelo.train()


    train_loss_total = 0

    numero_batches = 0


    # --------------------------------------------------
    # MINI-BATCH TRAINING
    # --------------------------------------------------

    for X_batch, y_batch in train_loader:

        X_batch = X_batch.to(
            device
        )

        y_batch = y_batch.to(
            device
        )


        loss = calcular_loss_batch(
            X_batch,
            y_batch
        )


        optimizer.zero_grad()


        loss.backward()


        optimizer.step()


        train_loss_total += (
            loss.item()
        )


        numero_batches += 1


    train_loss = (
        train_loss_total
        /
        numero_batches
    )


    # --------------------------------------------------
    # VALIDATION
    # --------------------------------------------------

    if epoch % evaluar_cada == 0:

        validation_loss = evaluar(
            val_loader
        )


        print(

            f"Epoch {epoch:4d} | "

            f"Train Loss: "
            f"{train_loss:.4f} | "

            f"Validation Loss: "
            f"{validation_loss:.4f}"

        )


        # --------------------------------------------------
        # ¿MEJORÓ VALIDATION?
        # --------------------------------------------------

        if (
            validation_loss
            <
            mejor_validation_loss
        ):

            mejor_validation_loss = (
                validation_loss
            )


            mejor_modelo = copy.deepcopy(
                modelo.state_dict()
            )


            sin_mejorar = 0


        else:

            sin_mejorar += 1


        # --------------------------------------------------
        # EARLY STOPPING
        # --------------------------------------------------

        if sin_mejorar >= patience:

            print(
                "\nEarly stopping activado."
            )

            print(
                "Validation dejó de mejorar."
            )

            break


# --------------------------------------------------
# RESTAURAR MEJOR MODELO
# --------------------------------------------------

if mejor_modelo is not None:

    modelo.load_state_dict(
        mejor_modelo
    )


# --------------------------------------------------
# RESULTADOS FINALES
# --------------------------------------------------

train_final = evaluar(
    train_loader
)


validation_final = evaluar(
    val_loader
)


print("\nResultado final:")


print(
    "Train Loss:",
    train_final
)


print(
    "Validation Loss:",
    validation_final
)


print(
    "\nMejor Validation Loss:",
    mejor_validation_loss
)