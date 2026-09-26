texto = """
el perro come
el gato come
el perro duerme
el gato duerme
"""

tokens = texto.split()

print("Tokens:")
print(tokens)


# --------------------------------------------------
# VOCABULARIO
# --------------------------------------------------

vocabulario = sorted(set(tokens))

token_a_id = {
    token: indice
    for indice, token in enumerate(vocabulario)
}

id_a_token = {
    indice: token
    for token, indice in token_a_id.items()
}


print("\nVocabulario:")
print(vocabulario)


# --------------------------------------------------
# CREAR PARES
# token actual -> siguiente token
# --------------------------------------------------

pares = []

for i in range(len(tokens) - 1):

    entrada = tokens[i]
    objetivo = tokens[i + 1]

    pares.append(
        (
            token_a_id[entrada],
            token_a_id[objetivo]
        )
    )


print("\nPares de entrenamiento:\n")

for entrada_id, objetivo_id in pares:

    entrada = id_a_token[entrada_id]
    objetivo = id_a_token[objetivo_id]

    print(
        f"{entrada:8} -> {objetivo}"
    )