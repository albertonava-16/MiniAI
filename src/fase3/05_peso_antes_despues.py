import torch
import torch.nn as nn


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Usando:", device)


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

criterio = nn.MSELoss()

optimizer = torch.optim.SGD(
    modelo.parameters(),
    lr=0.5
)


# --------------------------------------------------
# PESO ANTES
# --------------------------------------------------

peso_antes = modelo.capa1.weight[0, 0].item()

print("\nPeso antes del entrenamiento:")
print(peso_antes)


# --------------------------------------------------
# FORWARD
# --------------------------------------------------

output = modelo(X)

loss = criterio(output, y)

print("\nLoss:")
print(loss.item())


# --------------------------------------------------
# BACKPROPAGATION
# --------------------------------------------------

optimizer.zero_grad()

loss.backward()


# --------------------------------------------------
# GRADIENTE
# --------------------------------------------------

gradiente = modelo.capa1.weight.grad[0, 0].item()

print("\nGradiente de ese peso:")
print(gradiente)


# --------------------------------------------------
# ACTUALIZACIÓN
# --------------------------------------------------

optimizer.step()


# --------------------------------------------------
# PESO DESPUÉS
# --------------------------------------------------

peso_despues = modelo.capa1.weight[0, 0].item()

print("\nPeso después de optimizer.step():")
print(peso_despues)


# --------------------------------------------------
# CAMBIO
# --------------------------------------------------

print("\nCambio realizado:")
print(peso_despues - peso_antes)