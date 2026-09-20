import time
import torch
import torch.nn as nn


class RedXOR(nn.Module):

    def __init__(self):
        super().__init__()

        self.capa1 = nn.Linear(2, 2)
        self.capa2 = nn.Linear(2, 1)

    def forward(self, x):
        x = torch.sigmoid(self.capa1(x))
        x = torch.sigmoid(self.capa2(x))
        return x


def entrenar(device, epochs=10000):

    X = torch.tensor([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0]
    ], device=device)

    y = torch.tensor([
        [0.0],
        [1.0],
        [1.0],
        [0.0]
    ], device=device)

    modelo = RedXOR().to(device)

    criterio = nn.MSELoss()

    optimizer = torch.optim.SGD(
        modelo.parameters(),
        lr=0.5
    )

    if device.type == "cuda":
        torch.cuda.synchronize()

    inicio = time.perf_counter()

    for epoch in range(epochs):

        output = modelo(X)

        loss = criterio(output, y)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    if device.type == "cuda":
        torch.cuda.synchronize()

    fin = time.perf_counter()

    return fin - inicio, loss.item()


# ----------------------------
# CPU
# ----------------------------

cpu = torch.device("cpu")

tiempo_cpu, loss_cpu = entrenar(cpu)


# ----------------------------
# GPU
# ----------------------------

if torch.cuda.is_available():

    gpu = torch.device("cuda")

    tiempo_gpu, loss_gpu = entrenar(gpu)

    print("\n--- RESULTADOS ---\n")

    print(f"CPU:")
    print(f"Tiempo: {tiempo_cpu:.4f} segundos")
    print(f"Loss: {loss_cpu:.6f}")

    print()

    print(f"GPU:")
    print(f"Tiempo: {tiempo_gpu:.4f} segundos")
    print(f"Loss: {loss_gpu:.6f}")

else:

    print("CUDA no disponible")