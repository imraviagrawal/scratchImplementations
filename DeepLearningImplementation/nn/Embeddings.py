"""
Here we will implement different types of embeddings


"""
import math
import torch
from torch import nn
from torch import Tensor

# Todo: implement position embedding and create more understanding.
def position_encoding(
    seq_len: int, dim_model: int, device: torch.device = torch.device("cpu"),
) -> Tensor:
    pos = torch.arange(seq_len, dtype=torch.float, device=device).reshape(1, -1, 1)
    dim = torch.arange(dim_model, dtype=torch.float, device=device).reshape(1, 1, -1)
    phase = pos / (1e4 ** (dim / dim_model))

    return torch.where(dim.long() % 2 == 0, torch.sin(phase), torch.cos(phase))

# class PositionalEncoding(nn.Module):
#
#     def __init__(self, d_model, max_len=5000):
#         """
#         Inputs
#             d_model - Hidden dimensionality of the input.
#             max_len - Maximum length of a sequence to expect.
#         """
#         # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0)
#
#         # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
#         # Used for tensors that need to be on the same device as the module.
#         # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
#         self.register_buffer('pe', pe, persistent=False)
#
#     def forward(self, x):
#         x = x + self.pe[:, :x.size(1)]
#         return x
