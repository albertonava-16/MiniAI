import torch


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


# --------------------------------------------------
# TOKENIZAR
# --------------------------------------------------

tokens = texto.split()


vocabulario = sorted(set(tokens))


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
# CONFIGURACIÓN
# --------------------------------------------------

context_size = 4


# --------------------------------------------------
# CREAR EJEMPLOS
# --------------------------------------------------

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

        entradas.append(
            entrada
        )

        objetivos.append(
            objetivo
        )


    X = torch.stack(
        entradas
    )

    y = torch.stack(
        objetivos
    )

    return X, y


X_train, y_train = crear_ejemplos(
    train_data,
    context_size
)


X_val, y_val = crear_ejemplos(
    validation_data,
    context_size
)


# --------------------------------------------------
# MOSTRAR FORMAS
# --------------------------------------------------

print("Forma X_train:")
print(X_train.shape)

print("\nForma y_train:")
print(y_train.shape)

print("\nForma X_val:")
print(X_val.shape)

print("\nForma y_val:")
print(y_val.shape)


# --------------------------------------------------
# MOSTRAR ALGUNOS EJEMPLOS
# --------------------------------------------------

print("\nEjemplos de entrenamiento:\n")


for i in range(
    min(5, len(X_train))
):

    entrada_tokens = [
        id_a_token[id.item()]
        for id in X_train[i]
    ]

    objetivo_tokens = [
        id_a_token[id.item()]
        for id in y_train[i]
    ]

    print(
        "Entrada: ",
        entrada_tokens
    )

    print(
        "Objetivo:",
        objetivo_tokens
    )

    print()