import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Usando:", device)


# ============================================================
# CARGAR CHECKPOINT FASE 7
# ============================================================

checkpoint = torch.load(
    "../fase7/miniai_fase7_checkpoint.pth",
    map_location=device
)

print("\nCheckpoint Fase 7 cargado.")


# ============================================================
# CONFIGURACIÓN RECUPERADA
# ============================================================

config = checkpoint[
    "config"
]

context_size = config[
    "max_seq_len"
]

BOS = checkpoint[
    "special_tokens"
]["BOS"]

EOS = checkpoint[
    "special_tokens"
]["EOS"]

vocabulario = checkpoint[
    "vocabulario"
]

token_a_id = {
    token: i
    for i, token in enumerate(vocabulario)
}

id_a_token = {
    i: token
    for i, token in enumerate(vocabulario)
}


# ============================================================
# DATASET
# ============================================================

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
    "el gato mira al perro"
]


val_oraciones = [
    "el perro corre por el parque",
    "el gato duerme en la casa",
    "la niña juega en el parque",
    "el niño corre con el perro",
    "el perro mira al gato",
    "el gato mira al perro"
]


train_oraciones = [
    oracion
    for oracion in oraciones
    if oracion not in val_oraciones
]


# ============================================================
# CREAR EJEMPLOS
# ============================================================

def crear_ejemplos(oraciones):

    X = []
    y = []

    for oracion in oraciones:

        tokens = [
            BOS
        ] + oracion.split() + [
            EOS
        ]

        ids = [
            token_a_id[token]
            for token in tokens
        ]

        for i in range(
            1,
            len(ids)
        ):

            contexto = ids[
                max(
                    0,
                    i - context_size
                ):i
            ]

            while len(contexto) < context_size:

                contexto.insert(
                    0,
                    token_a_id[BOS]
                )

            objetivo = ids[i]

            X.append(
                contexto
            )

            y.append(
                objetivo
            )

    return (
        torch.tensor(
            X,
            dtype=torch.long
        ),
        torch.tensor(
            y,
            dtype=torch.long
        )
    )


X_train, y_train = crear_ejemplos(
    train_oraciones
)

X_val, y_val = crear_ejemplos(
    val_oraciones
)


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
    batch_size=8,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=8,
    shuffle=False
)


# ============================================================
# ATTENTION HEAD
# ============================================================

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


# ============================================================
# MULTI-HEAD ATTENTION
# ============================================================

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


# ============================================================
# FEED FORWARD
# ============================================================

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


# ============================================================
# TRANSFORMER BLOCK
# ============================================================

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


# ============================================================
# MINI GPT
# ============================================================

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
    
# ============================================================
# CREAR MODELO
# ============================================================

modelo = MiniGPT(
    vocab_size=len(
        vocabulario
    ),
    embedding_dim=config[
        "embedding_dim"
    ],
    num_heads=config[
        "num_heads"
    ],
    max_seq_len=config[
        "max_seq_len"
    ],
    num_blocks=config[
        "num_blocks"
    ],
    dropout=config[
        "dropout"
    ]
).to(
    device
)


# ============================================================
# CARGAR PESOS DE FASE 7
# ============================================================

modelo.load_state_dict(
    checkpoint[
        "model_state_dict"
    ]
)

print(
    "Pesos de Fase 7 cargados."
)


# ============================================================
# LOSS
# ============================================================

criterio = nn.CrossEntropyLoss()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    modelo.parameters(),
    lr=0.001,
    weight_decay=0.01
)


print(
    "\nModelo listo para continuar entrenamiento."
)

print(
    "Vocabulario:",
    len(
        vocabulario
    ),
    "tokens"
)

print(
    "Configuración:",
    config
)


# ============================================================
# VALIDATION LOSS
# ============================================================

def calcular_validation_loss(
    modelo,
    loader
):

    modelo.eval()

    total = 0.0

    with torch.no_grad():

        for X_batch, y_batch in loader:

            X_batch = X_batch.to(
                device
            )

            y_batch = y_batch.to(
                device
            )

            logits = modelo(
                X_batch
            )

            ultimo_logits = logits[
                :,
                -1,
                :
            ]

            loss = criterio(
                ultimo_logits,
                y_batch
            )

            total += loss.item()

    return (
        total
        /
        len(loader)
    )


# ============================================================
# MEDIR ANTES DE CONTINUAR
# ============================================================

validation_antes = calcular_validation_loss(
    modelo,
    val_loader
)

print(
    "\nValidation antes de continuar:",
    validation_antes
)


# ============================================================
# CONTINUAR ENTRENAMIENTO
# ============================================================

epochs_adicionales = 200

mejor_validation_loss = (
    validation_antes
)

mejor_modelo = copy.deepcopy(
    modelo.state_dict()
)

# Guardamos el optimizador en el mismo punto que los mejores pesos.
# Si no hay mejora, este estado corresponde a un AdamW nuevo,
# porque Fase 7 no guardo su optimizer_state_dict.
mejor_optimizer_state = copy.deepcopy(
    optimizer.state_dict()
)

# -1 indica el punto de partida, antes de entrenar nuevos epochs.
mejor_epoch = -1

sin_mejorar = 0

patience = 5


print(
    "\nContinuando entrenamiento...\n"
)


for epoch in range(
    epochs_adicionales
):

    modelo.train()

    train_loss_total = 0.0

    for X_batch, y_batch in train_loader:

        X_batch = X_batch.to(
            device
        )

        y_batch = y_batch.to(
            device
        )

        optimizer.zero_grad()

        logits = modelo(
            X_batch
        )

        ultimo_logits = logits[
            :,
            -1,
            :
        ]

        loss = criterio(
            ultimo_logits,
            y_batch
        )

        loss.backward()

        optimizer.step()

        train_loss_total += (
            loss.item()
        )


    if epoch % 20 == 0:

        train_loss = (
            train_loss_total
            /
            len(
                train_loader
            )
        )

        validation_loss = (
            calcular_validation_loss(
                modelo,
                val_loader
            )
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

            mejor_optimizer_state = copy.deepcopy(
                optimizer.state_dict()
            )

            mejor_epoch = epoch

            sin_mejorar = 0

        else:

            sin_mejorar += 1


        if (
            sin_mejorar
            >=
            patience
        ):

            print(
                "\nEarly stopping."
            )

            break


# ============================================================
# RESTAURAR MEJOR MODELO
# ============================================================

modelo.load_state_dict(
    mejor_modelo
)

# Restauramos tambien el optimizador del mismo instante.
optimizer.load_state_dict(
    mejor_optimizer_state
)

modelo.eval()


print(
    "\nValidation ANTES:",
    validation_antes
)

print(
    "Mejor Validation DESPUÉS:",
    mejor_validation_loss
)

print("Mejor epoch adicional:", mejor_epoch)
print("Último epoch ejecutado:", epoch)


# ============================================================
# GUARDAR CHECKPOINT DE ENTRENAMIENTO
# ============================================================

checkpoint_entrenamiento = {

    "model_state_dict":
        modelo.state_dict(),

    "optimizer_state_dict":
        optimizer.state_dict(),

    "epoch":
        mejor_epoch,

    "config":
        config,

    "vocabulario":
        vocabulario,

    "special_tokens": {
        "BOS": BOS,
        "EOS": EOS
    },

    "training": {

        "optimizer":
            "AdamW",

        "learning_rate":
            0.001,

        "weight_decay":
            0.01,

        "best_validation_loss":
            mejor_validation_loss,

        "best_epoch":
            mejor_epoch,

        "last_epoch_executed":
            epoch,

        "optimizer_from_phase7_restored":
            False,

        "sin_mejorar":
            sin_mejorar,

        "patience":
            patience
    }
}


torch.save(
    checkpoint_entrenamiento,
    "miniai_training_checkpoint.pth"
)


print(
    "\nCheckpoint de entrenamiento guardado:"
)

print(
    "miniai_training_checkpoint.pth"
)