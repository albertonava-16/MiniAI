import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(output):
    return output * (1 - output)


# --------------------------------------------------
# Usamos los pesos que tu red ya aprendió
# --------------------------------------------------

W1 = np.array([
    [-6.51540735,  6.69881652],
    [ 6.70947189, -6.49367016]
])

b1 = np.array([
    [3.29197597, 3.28197114]
])

W2 = np.array([
    [-8.29241772],
    [-8.29443745]
])

b2 = np.array([
    [12.04524094]
])


# --------------------------------------------------
# Elegimos un solo ejemplo
# XOR: [0, 1] debería dar 1
# --------------------------------------------------

entrada = np.array([[0.0, 1.0]])
esperado = np.array([[1.0]])


# ==================================================
# FORWARD PROPAGATION
# ==================================================

hidden_input = entrada @ W1 + b1
hidden_output = sigmoid(hidden_input)

final_input = hidden_output @ W2 + b2
output = sigmoid(final_input)


print("\n--- FORWARD ---\n")

print("Entrada:")
print(entrada)

print("\nSalida capa oculta:")
print(hidden_output)

print("\nSalida final:")
print(output)


# ==================================================
# ERROR
# ==================================================

error = esperado - output


print("\n--- ERROR ---\n")

print("Esperado:")
print(esperado)

print("\nObtenido:")
print(output)

print("\nError:")
print(error)


# ==================================================
# BACKPROPAGATION
# ==================================================

output_delta = error * sigmoid_derivative(output)

hidden_error = output_delta @ W2.T

hidden_delta = hidden_error * sigmoid_derivative(hidden_output)


print("\n--- BACKPROPAGATION ---\n")

print("1. output_delta:")
print(output_delta)

print("\n2. hidden_error:")
print(hidden_error)

print("\n3. hidden_delta:")
print(hidden_delta)