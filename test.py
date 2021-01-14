import torch

a = torch.rand(3, 4, requires_grad=True)
b = torch.rand(3, 4, requires_grad=True)

c = torch.nn.Linear(4, 10)(a * b)
print(c)

res = []
for i in range(len(c)):
    res.append((c[i] * 2).unsqueeze(0))

res = torch.cat(res, dim=0)

c.retain_grad()
res.backward()


print()

