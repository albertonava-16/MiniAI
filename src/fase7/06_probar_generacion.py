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


context_size = 4


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
            for _ in range(num_blocks)
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

        return self.salida(x)


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

    logits = logits.reshape(
        -1,
        len(vocabulario)
    )

    targets = y.reshape(
        -1
    )

    return criterio(
        logits,
        targets
    )


# --------------------------------------------------
# VALIDATION
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

mejor_validation_loss = float(
    "inf"
)

mejor_modelo = None

patience = 8

sin_mejorar = 0

evaluar_cada = 20


print("Entrenando...\n")


for epoch in range(epochs):

    modelo.train()

    train_loss_total = 0
    numero_batches = 0


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


    if epoch % evaluar_cada == 0:

        validation_loss = evaluar(
            val_loader
        )

        print(
            f"Epoch {epoch:4d} | "
            f"Train: {train_loss:.4f} | "
            f"Validation: {validation_loss:.4f}"
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
# RESTAURAR MEJOR MODELO
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
# GENERACIÓN
# --------------------------------------------------

def generar(
    prompt,
    tokens_nuevos=4
):

    palabras = prompt.split()


    for palabra in palabras:

        if palabra not in token_a_id:

            raise ValueError(
                f"Token desconocido: {palabra}"
            )


    ids = [
        token_a_id[palabra]
        for palabra in palabras
    ]


    for _ in range(tokens_nuevos):

        # Solo podemos usar como máximo
        # context_size tokens.

        contexto = ids[
            -context_size:
        ]


        entrada = torch.tensor(
            [contexto],
            dtype=torch.long,
            device=device
        )


        with torch.no_grad():

            logits = modelo(
                entrada
            )


        # Predicción correspondiente
        # al último token.

        ultimo_logits = logits[
            0,
            -1
        ]


        probabilidades = F.softmax(
            ultimo_logits,
            dim=0
        )


        siguiente_id = torch.argmax(
            probabilidades
        ).item()


        ids.append(
            siguiente_id
        )


    resultado = [
        id_a_token[id]
        for id in ids
    ]


    return " ".join(
        resultado
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


print("\n--- GENERACIÓN ---\n")


for prompt in prompts:

    resultado = generar(
        prompt,
        tokens_nuevos=4
    )

    print(
        f"{prompt:16} -> {resultado}"
    )