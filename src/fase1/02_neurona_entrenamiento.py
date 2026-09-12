import math


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


# Pesos iniciales
w1 = 0.8
w2 = 0.4
bias = -0.5

# Qué tan grandes serán nuestros ajustes
learning_rate = 0.5


datos = [
    (0, 0, 0),
    (0, 1, 0),
    (1, 0, 0),
    (1, 1, 1)
]


for epoch in range(10000):

    for x1, x2, esperado in datos:

        # 1. La neurona calcula
        z = x1 * w1 + x2 * w2 + bias

        output = sigmoid(z)

        # 2. Calculamos cuánto se equivocó
        error = esperado - output

        # 3. Calculamos cuánto debemos corregir
        delta = error * output * (1 - output)

        # 4. Ajustamos los pesos
        w1 = w1 + learning_rate * delta * x1
        w2 = w2 + learning_rate * delta * x2

        # 5. Ajustamos el bias
        bias = bias + learning_rate * delta


print("Entrenamiento terminado")
print()

print("w1:", w1)
print("w2:", w2)
print("bias:", bias)

print()
print("Resultados:")
print()


for x1, x2, esperado in datos:

    z = x1 * w1 + x2 * w2 + bias

    output = sigmoid(z)

    prediccion = 1 if output >= 0.5 else 0

    print(
        f"Entrada: ({x1}, {x2}) "
        f"Esperado: {esperado} "
        f"Salida: {output:.4f} "
        f"Predicción: {prediccion}"
    )