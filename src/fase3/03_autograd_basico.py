import torch

x = torch.tensor(3.0, requires_grad=True)

y = x ** 2 + 2 * x + 1

print("x:", x)
print("y:", y)

y.backward()

print("Gradiente de x:", x.grad)