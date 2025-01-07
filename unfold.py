import torch
import torch.nn.functional as F

x = torch.arange(16).view(1, 1, 4, 4).float()

print(x)

x = F.unfold(x, 3)
print(x)
print(x.size())