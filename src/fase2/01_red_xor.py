import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    # Convierte cualquier número en un valor entre 0 y 1.
    # sigmoid(0) = 0.5
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(output):
    # Calcula la pendiente de sigmoid.
    # Nos ayuda a saber cuánto corregir durante el entrenamiento.
    return output * (1 - output)


# ==========================================================
# DATASET XOR
# ==========================================================

# XOR vale 1 cuando las entradas son diferentes.
# XOR vale 0 cuando las entradas son iguales.

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], dtype=float)

# Respuestas correctas

y = np.array([
    [0],
    [1],
    [1],
    [0]
], dtype=float)


# ==========================================================
# PESOS INICIALES
# ==========================================================

# La semilla permite obtener los mismos números aleatorios
# cada vez que ejecutamos el programa.
np.random.seed(42)

# 2 entradas -> 2 neuronas ocultas
W1 = np.random.uniform(-1, 1, (2, 2))

# Bias de las 2 neuronas ocultas
b1 = np.zeros((1, 2))

# 2 neuronas ocultas -> 1 neurona de salida
W2 = np.random.uniform(-1, 1, (2, 1))

# Bias de la neurona de salida
b2 = np.zeros((1, 1))


learning_rate = 0.5
epochs = 10000


print("\nPesos iniciales W1:")
print(W1)

print("\nPesos iniciales W2:")
print(W2)

loss_history = []

# ==========================================================
# ENTRENAMIENTO
# ==========================================================

for epoch in range(epochs):

    # --------------------------
    # FORWARD PROPAGATION
    # --------------------------

    # Entrada -> capa oculta
    hidden_input = X @ W1 + b1

    hidden_output = sigmoid(hidden_input)

    # Capa oculta -> salida
    final_input = hidden_output @ W2 + b2

    output = sigmoid(final_input)


    # --------------------------
    # ERROR
    # --------------------------

    error = y - output

    loss = np.mean(error ** 2)
    loss_history.append(loss)

    


    # --------------------------
    # BACKPROPAGATION
    # --------------------------

    # Error de la capa de salida
    output_delta = error * sigmoid_derivative(output)

    # Propagamos el error hacia la capa oculta
    hidden_error = output_delta @ W2.T

    hidden_delta = hidden_error * sigmoid_derivative(hidden_output)


    # --------------------------
    # ACTUALIZACIÓN DE PESOS
    # --------------------------

    W2 += learning_rate * hidden_output.T @ output_delta

    b2 += learning_rate * np.sum(
        output_delta,
        axis=0,
        keepdims=True
    )

    W1 += learning_rate * X.T @ hidden_delta

    b1 += learning_rate * np.sum(
        hidden_delta,
        axis=0,
        keepdims=True
    )


# ==========================================================
# AQUÍ YA TERMINÓ EL ENTRENAMIENTO
# ==========================================================

# ==========================================================
# GRAFICA DEL LOSS
# ==========================================================

import matplotlib.pyplot as plt

valores = [10, 8, 6, 4, 2, 1]

plt.plot(valores)
plt.title("Prueba de Matplotlib")
plt.xlabel("Paso")
plt.ylabel("Valor")
plt.grid()

plt.savefig("prueba_matplotlib.png")

print("Se intentó guardar prueba_matplotlib.png")

plt.show()

print("\n--- RECORRIDO DE UNA ENTRADA ---\n")


# Vamos a seguir solamente la entrada [0, 1]

entrada = np.array([0, 1], dtype=float)


print("Entrada original:")
print(entrada)


# PASO 1
# Combinamos la entrada con pesos y bias

hidden_input = entrada @ W1 + b1

print("\n1. Resultado antes de sigmoid en capa oculta:")
print(hidden_input)


# PASO 2
# Las dos neuronas ocultas se activan

hidden_output = sigmoid(hidden_input)

print("\n2. Salida de las neuronas ocultas:")
print(hidden_output)


# PASO 3
# La neurona de salida recibe las dos activaciones anteriores

final_input = hidden_output @ W2 + b2

print("\n3. Resultado antes de sigmoid en salida:")
print(final_input)


# PASO 4
# Convertimos el resultado a un valor entre 0 y 1

final_output = sigmoid(final_input)

print("\n4. Salida final:")
print(final_output)


# PASO 5
# Convertimos la salida decimal en una clase

prediccion = 1 if final_output[0, 0] >= 0.5 else 0

print("\n5. Predicción:")
print(prediccion)


# ==========================================================
# QUÉ PRODUCE LA CAPA OCULTA
# ==========================================================

print("\n--- SALIDAS DE LA CAPA OCULTA ---\n")

for entrada in X:

    hidden_input = entrada @ W1 + b1

    hidden_output = sigmoid(hidden_input)

    print(
        f"Entrada: {entrada.astype(int)} "
        f"-> Neurona oculta 1: {hidden_output[0, 0]:.4f} "
        f"Neurona oculta 2: {hidden_output[0, 1]:.4f}"
    )


# ==========================================================
# RESULTADOS FINALES
# ==========================================================

# Volvemos a calcular todo usando los pesos ya aprendidos

hidden_output = sigmoid(X @ W1 + b1)

output = sigmoid(hidden_output @ W2 + b2)


print("\n--- RESULTADOS FINALES ---\n")


print("Pesos finales W1:")
print(W1)

print("\nPesos finales W2:")
print(W2)

print()


for entrada, esperado, salida in zip(X, y, output):

    prediccion = 1 if salida[0] >= 0.5 else 0

    print(
        f"Entrada: {entrada.astype(int)} "
        f"Esperado: {int(esperado[0])} "
        f"Salida: {salida[0]:.4f} "
        f"Predicción: {prediccion}"
    )