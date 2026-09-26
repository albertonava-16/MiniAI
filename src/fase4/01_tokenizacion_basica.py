texto = "hola mundo hola ia"

# 1. Tokenización muy simple por espacios
tokens = texto.split()

print("Texto original:")
print(texto)

print("\nTokens:")
print(tokens)


# 2. Crear vocabulario único
vocabulario = sorted(set(tokens))

print("\nVocabulario:")
print(vocabulario)


# 3. Asignar un ID a cada token
token_a_id = {
    token: indice
    for indice, token in enumerate(vocabulario)
}

print("\nToken -> ID:")
print(token_a_id)


# 4. Convertir texto a IDs
ids = [
    token_a_id[token]
    for token in tokens
]

print("\nTexto convertido a IDs:")
print(ids)