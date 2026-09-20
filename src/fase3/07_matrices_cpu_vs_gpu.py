import time
import torch


def medir_cpu(size=3000, repeticiones=20):

    a = torch.randn(size, size)
    b = torch.randn(size, size)

    inicio = time.perf_counter()

    for _ in range(repeticiones):
        c = a @ b

    fin = time.perf_counter()

    return fin - inicio


def medir_gpu(size=3000, repeticiones=20):

    device = torch.device("cuda")

    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)

    # Warm-up:
    # hacemos una operación antes de medir para que CUDA ya esté "despierto"
    c = a @ b
    torch.cuda.synchronize()

    inicio = time.perf_counter()

    for _ in range(repeticiones):
        c = a @ b

    torch.cuda.synchronize()

    fin = time.perf_counter()

    return fin - inicio


print("Midiendo CPU...")
tiempo_cpu = medir_cpu()

print("Midiendo GPU...")
tiempo_gpu = medir_gpu()

print("\n--- RESULTADOS ---\n")

print(f"CPU: {tiempo_cpu:.4f} segundos")
print(f"GPU: {tiempo_gpu:.4f} segundos")

print()

if tiempo_gpu > 0:
    aceleracion = tiempo_cpu / tiempo_gpu
    print(f"GPU aproximadamente {aceleracion:.2f}x más rápida")