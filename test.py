import torch

print(torch.__version__)
print(torch.cuda.is_available())

x = torch.tensor([1, 2, 3])
print(x)