import matplotlib.pyplot as plt
import numpy as np

# Pesos aprendidos
w1 = 7.2855553237377535
w2 = 7.285220290205796
bias = -11.014982989862922

# Datos AND
x1 = [0, 0, 1, 1]
x2 = [0, 1, 0, 1]
clases = [0, 0, 0, 1]

# Dibujar puntos
for i in range(len(x1)):
    plt.scatter(x1[i], x2[i], s=100)

    plt.text(
        x1[i] + 0.03,
        x2[i] + 0.03,
        f"Clase {clases[i]}"
    )

# Frontera de decisión:
# w1*x1 + w2*x2 + bias = 0
#
# despejamos x2:
#
# x2 = -(w1*x1 + bias) / w2

x = np.linspace(-0.2, 1.2, 100)

y = -(w1 * x + bias) / w2

plt.plot(x, y)

plt.xlim(-0.2, 1.2)
plt.ylim(-0.2, 1.2)

plt.xlabel("x1")
plt.ylabel("x2")

plt.title("Frontera de decisión aprendida para AND")

plt.grid()

plt.show()