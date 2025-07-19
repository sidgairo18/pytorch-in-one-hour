import torch
import torch.nn.functional as F
from torch.autograd import grad

y = torch.tensor([1.0])
x1 = torch.tensor([1.1])
w1 = torch.tensor([2.2], requires_grad=True)
b = torch.tensor([0.0], requires_grad=True)

z = x1*w1 + b
a = torch.sigmoid(z)

loss = F.binary_cross_entropy(a, y)

# using loss.backward()
"""
loss.backward()
print(w1.grad)
print(b.grad)
"""

grad_L_w1 = grad(loss, w1, retain_graph=True)
grad_L_b = grad(loss, b, retain_graph=True)

print("Loss w.r.t. w1: ", grad_L_w1)
print("Loss w.r.t. bias: ", grad_L_b)
