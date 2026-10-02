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


print("Cantidad de tokens:")
print(len(tokens))


# --------------------------------------------------
# VOCABULARIO
# --------------------------------------------------

vocabulario = sorted(set(tokens))


print("\nTamaño del vocabulario:")
print(len(vocabulario))


print("\nVocabulario:")
print(vocabulario)


# --------------------------------------------------
# TOKEN -> ID
# --------------------------------------------------

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


# --------------------------------------------------
# TEXTO -> IDS
# --------------------------------------------------

ids = [
    token_a_id[token]
    for token in tokens
]


datos = torch.tensor(
    ids,
    dtype=torch.long
)


print("\nPrimeros tokens:")
print(tokens[:20])


print("\nPrimeros IDs:")
print(datos[:20])


# --------------------------------------------------
# TRAIN / VALIDATION
# --------------------------------------------------

division = int(
    len(datos) * 0.8
)


train_data = datos[:division]

validation_data = datos[division:]


print("\nTokens entrenamiento:")
print(len(train_data))


print("\nTokens validación:")
print(len(validation_data))