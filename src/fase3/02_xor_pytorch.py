import torch
import torch.nn as nn


# --------------------------------------------------
# DISPOSITIVO
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Usando:", device)


# --------------------------------------------------
# DATOS XOR
# --------------------------------------------------

X = torch.tensor([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
]).to(device)

y = torch.tensor([
    [0.0],
    [1.0],
    [1.0],
    [0.0]
]).to(device)


# --------------------------------------------------
# MODELO
# --------------------------------------------------

class RedXOR(nn.Module):

    def __init__(self):
        super().__init__()

        self.capa1 = nn.Linear(2, 2)
        self.capa2 = nn.Linear(2, 1)

    def forward(self, x):

        x = torch.sigmoid(self.capa1(x))

        x = torch.sigmoid(self.capa2(x))

        return x


modelo = RedXOR().to(device)


# --------------------------------------------------
# LOSS
# --------------------------------------------------

criterio = nn.MSELoss()


# --------------------------------------------------
# OPTIMIZER
# --------------------------------------------------

optimizer = torch.optim.SGD(
    modelo.parameters(),
    lr=0.5
)


# --------------------------------------------------
# ENTRENAMIENTO
# --------------------------------------------------

epochs = 10000

for epoch in range(epochs):

    output = modelo(X)

    loss = criterio(output, y)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()


# --------------------------------------------------
# RESULTADOS
# --------------------------------------------------

print("\nLoss final:", loss.item())

with torch.no_grad():

    output = modelo(X)

    for entrada, esperado, salida in zip(X, y, output):

        prediccion = 1 if salida.item() >= 0.5 else 0

        print(
            f"Entrada: {entrada.tolist()} "
            f"Esperado: {int(esperado.item())} "
            f"Salida: {salida.item():.4f} "
            f"Predicción: {prediccion}"
        )