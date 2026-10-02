import random


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


# --------------------------------------------------
# MEZCLAR
# --------------------------------------------------

random.seed(42)

random.shuffle(oraciones)


# --------------------------------------------------
# TRAIN / VALIDATION
# --------------------------------------------------

division = int(
    len(oraciones) * 0.8
)

train_oraciones = oraciones[:division]

val_oraciones = oraciones[division:]


print("Oraciones totales:")
print(len(oraciones))

print("\nTrain:")
print(len(train_oraciones))

print("\nValidation:")
print(len(val_oraciones))


print("\n--- Ejemplos TRAIN ---")

for frase in train_oraciones[:5]:
    print(frase)


print("\n--- VALIDATION ---")

for frase in val_oraciones:
    print(frase)


# --------------------------------------------------
# VOCABULARIO
# --------------------------------------------------

tokens = []

for frase in oraciones:
    tokens.extend(
        frase.split()
    )


vocabulario = sorted(
    set(tokens)
)


print("\nTamaño vocabulario:")
print(len(vocabulario))

print("\nVocabulario:")
print(vocabulario)