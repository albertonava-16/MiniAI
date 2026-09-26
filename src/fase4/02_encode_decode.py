texto = "hola mundo hola ia"

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


# ---------------------------------------
# ENCODE
# texto -> IDs
# ---------------------------------------

ids = [
    token_a_id[token]
    for token in tokens
]

print("Texto original:")
print(texto)

print("\nIDs:")
print(ids)


# ---------------------------------------
# DECODE
# IDs -> texto
# ---------------------------------------

tokens_recuperados = [
    id_a_token[id]
    for id in ids
]

texto_recuperado = " ".join(tokens_recuperados)

print("\nTokens recuperados:")
print(tokens_recuperados)

print("\nTexto recuperado:")
print(texto_recuperado)