import torch

# create a 0D tensor (scalar) from a Python integer
tensor0d = torch.tensor(1)

# create a 1D tensor (vector) from a Python list
tensor1d = torch.tensor([1, 2, 3])

# create a 2D tensor from a nested Python list
tensor2d = torch.tensor([[1, 2], [3, 4]])

# create a 3D tensor from a nested Python list
tensor3d = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

print(tensor0d)
print(tensor1d)
print(tensor2d)
print(tensor3d)

tensor2d = torch.tensor([[1,2,3], [4,5,6]])
print(tensor2d.shape)
