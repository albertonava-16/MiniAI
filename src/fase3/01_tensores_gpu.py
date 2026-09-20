import torch

print("PyTorch:", torch.__version__)
print("CUDA disponible:", torch.cuda.is_available())

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Dispositivo elegido:", device)

x = torch.tensor([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
])

print("\nTensor original:")
print(x)

print("\nDispositivo original:")
print(x.device)

x = x.to(device)

print("\nTensor después de moverlo:")
print(x)

print("\nDispositivo final:")
print(x.device)