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
# UNA SOLA PASADA
# --------------------------------------------------

output = modelo(X)

loss = criterio(output, y)

print("\nLoss antes de backward:")
print(loss.item())


# --------------------------------------------------
# LIMPIAMOS GRADIENTES
# --------------------------------------------------

modelo.zero_grad()


# --------------------------------------------------
# BACKPROPAGATION
# --------------------------------------------------

loss.backward()


# --------------------------------------------------
# MOSTRAMOS LOS GRADIENTES
# --------------------------------------------------

print("\n--- GRADIENTES ---\n")

print("Gradiente pesos capa 1:")
print(modelo.capa1.weight.grad)

print("\nGradiente bias capa 1:")
print(modelo.capa1.bias.grad)

print("\nGradiente pesos capa 2:")
print(modelo.capa2.weight.grad)

print("\nGradiente bias capa 2:")
print(modelo.capa2.bias.grad)