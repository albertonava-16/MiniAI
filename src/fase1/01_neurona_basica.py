import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

w1 = 0.8
w2 = 0.4
bias = -0.5

datos = [
    (0, 0, 0),
    (0, 1, 0),
    (1, 0, 0),
    (1, 1, 1)
]

for x1, x2, esperado in datos:
    z = x1 * w1 + x2 * w2 + bias
    output = sigmoid(z)

    prediccion = 1 if output >= 0.5 else 0

    error = esperado - output
    loss = error ** 2

    print(
        f"Entrada: ({x1}, {x2}) "
        f"Esperado: {esperado} "
        f"Salida: {output:.4f} "
        f"Predicción: {prediccion} "
        f"Loss: {loss:.4f}"
    )