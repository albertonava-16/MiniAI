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
# TOKENS ESPECIALES
# --------------------------------------------------

BOS = "<BOS>"
EOS = "<EOS>"


# --------------------------------------------------
# DATASET
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


# --------------------------------------------------
# AÑADIR BOS / EOS
# --------------------------------------------------

def agregar_tokens_especiales(frase):

    return (
        f"{BOS} "
        f"{frase} "
        f"{EOS}"
    )


train_oraciones = [
    agregar_tokens_especiales(frase)
    for frase in train_oraciones
]


val_oraciones = [
    agregar_tokens_especiales(frase)
    for frase in val_oraciones
]


# --------------------------------------------------
# VOCABULARIO
# --------------------------------------------------

tokens_totales = []


for frase in (
    train_oraciones
    +
    val_oraciones
):

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


print("\nVocabulario:")
print(vocabulario)


print(
    "\nBOS ID:",
    token_a_id[BOS]
)

print(
    "EOS ID:",
    token_a_id[EOS]
)


# --------------------------------------------------
# CONFIG
# --------------------------------------------------

context_size = 6


# --------------------------------------------------
# CREAR EJEMPLOS
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


        for i in range(
            len(ids) - 1
        ):

            inicio = max(
                0,
                i - context_size + 1
            )

            entrada = ids[
                inicio:i + 1
            ]

            objetivo = ids[
                i + 1
            ]


            # Padding simple al inicio
            # usando BOS

            while len(entrada) < context_size:

                entrada.insert(
                    0,
                    token_a_id[BOS]
                )


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

print("X_val:")
print(X_val.shape)


# --------------------------------------------------
# DATALOADERS
# --------------------------------------------------

train_loader = DataLoader(

    TensorDataset(
        X_train,
        y_train
    ),

    batch_size=8,

    shuffle=True
)


val_loader = DataLoader(

    TensorDataset(
        X_val,
        y_val
    ),

    batch_size=8,

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

        Q = self.W_Q(x)

        K = self.W_K(x)

        V = self.W_V(x)


        scores = (
            Q @ K.transpose(-2, -1)
        ) / math.sqrt(
            self.head_dim
        )


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


        return pesos @ V


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

            for _ in range(
                num_heads
            )

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


        return self.dropout(
            salida
        )


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


        x = self.norm1(
            x + attention_output
        )


        ff_output = self.feed_forward(
            x
        )


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


criterio = nn.CrossEntropyLoss()


optimizer = torch.optim.AdamW(

    modelo.parameters(),

    lr=0.003,

    weight_decay=0.01
)


# --------------------------------------------------
# LOSS
# --------------------------------------------------

def calcular_loss_batch(
    X,
    y
):

    logits = modelo(X)


    # Solo usamos la salida
    # correspondiente al último token.

    ultimo_logits = logits[
        :,
        -1,
        :
    ]


    return criterio(
        ultimo_logits,
        y
    )


# --------------------------------------------------
# EVALUAR
# --------------------------------------------------

def evaluar(loader):

    modelo.eval()

    total = 0

    batches = 0


    with torch.no_grad():

        for X, y in loader:

            X = X.to(device)

            y = y.to(device)


            loss = calcular_loss_batch(
                X,
                y
            )


            total += loss.item()

            batches += 1


    return total / batches


# --------------------------------------------------
# TRAIN
# --------------------------------------------------

epochs = 500

mejor_validation_loss = float(
    "inf"
)

mejor_modelo = None

sin_mejorar = 0

patience = 8

evaluar_cada = 20


print("\nEntrenando...\n")


for epoch in range(epochs):

    modelo.train()

    train_total = 0

    batches = 0


    for X, y in train_loader:

        X = X.to(device)

        y = y.to(device)


        loss = calcular_loss_batch(
            X,
            y
        )


        optimizer.zero_grad()

        loss.backward()

        optimizer.step()


        train_total += loss.item()

        batches += 1


    train_loss = (
        train_total / batches
    )


    if epoch % evaluar_cada == 0:

        validation_loss = evaluar(
            val_loader
        )


        print(

            f"Epoch {epoch:4d} | "

            f"Train: {train_loss:.4f} | "

            f"Validation: "
            f"{validation_loss:.4f}"
        )


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


        if sin_mejorar >= patience:

            print(
                "\nEarly stopping."
            )

            break


# --------------------------------------------------
# MEJOR MODELO
# --------------------------------------------------

modelo.load_state_dict(
    mejor_modelo
)

modelo.eval()


print(
    "\nMejor Validation Loss:",
    mejor_validation_loss
)


# --------------------------------------------------
# GENERAR HASTA EOS
# --------------------------------------------------

def generar(
    prompt,
    max_tokens=10
):

    palabras = prompt.split()


    ids = [
        token_a_id[BOS]
    ]


    for palabra in palabras:

        if palabra not in token_a_id:

            raise ValueError(
                f"Token desconocido: "
                f"{palabra}"
            )


        ids.append(
            token_a_id[palabra]
        )


    for _ in range(max_tokens):

        contexto = ids[
            -context_size:
        ]


        while len(contexto) < context_size:

            contexto.insert(
                0,
                token_a_id[BOS]
            )


        entrada = torch.tensor(

            [contexto],

            dtype=torch.long,

            device=device
        )


        with torch.no_grad():

            logits = modelo(
                entrada
            )


        ultimo_logits = logits[
            0,
            -1
        ]


        siguiente_id = torch.argmax(
            ultimo_logits
        ).item()


        # ------------------------------------------
        # SI GENERA EOS, TERMINAMOS
        # ------------------------------------------

        if siguiente_id == token_a_id[EOS]:

            break


        ids.append(
            siguiente_id
        )


    palabras_generadas = [

        id_a_token[id]

        for id in ids

        if id not in [

            token_a_id[BOS],

            token_a_id[EOS]
        ]
    ]


    return " ".join(
        palabras_generadas
    )


# --------------------------------------------------
# PRUEBAS
# --------------------------------------------------

prompts = [

    "el perro",

    "el gato",

    "la niña",

    "el niño",

    "el perro juega",

    "la niña corre",
]


print(
    "\n--- GENERACIÓN CON EOS ---\n"
)


for prompt in prompts:

    print(

        f"{prompt:16} -> "

        f"{generar(prompt)}"

    )