import torch
import torch as tc
# Assuming you have two tensors tensor1 and tensor2 with shapes (bs, a, 1)
# Create some sample tensors for demonstration
bs = 2  # batch size
a = 3   # dimension 'a'

# Sample tensors
t1 = torch.randn(bs, a, 1)
t2 = torch.randn(bs, a, 1)

# Concatenate along the last dimension
t3 = torch.cat((t1, t2), dim=2)

# Shape of concatenated tensor
print(t3.shape)  # Output: torch.Size([bs, a, 2])
t4 = t3.reshape(bs,-1)
breakpoint()