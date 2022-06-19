import torch

a = torch.ones(3)
print(a)

b = torch.zeros(3)
print(b)

k = torch.tensor([1.0, 3.0, 6.7])
print(k)

points = torch.tensor([[1.0, 4.0], [2.0, 1.0], [3.0, 5.0]])
print(points)
print(points.shape)
points.storage()
