# this file will get all shared modules scratch implementations for modules
"""
Multi Head Attention will come here

Query: Its a feature vector which describes what we are looking for in the input sequence.
Key: Key is a feature vector which describes what this element is offering or when it may be important.
Values: For each input element, we also have a value vector. This feature vector is the one we want to average over.

Attention = softmax(Qk^T/sqrt(dk))*V
Head = Attention(QWq, KWk, VWv) # Wx is shared across heads, where x is q, k, v
Multi-head = concat(head1, head2....)*Wo
"""

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F # imports functions

def scaled_dot_attention(q, k, v, mask=None):
    """

    :param q: batch, n_token, q_dim
    :param k: batch, n_token, k_dim
    :param v: batch, n_token, v_dim
    :return:
    """
    d_k = q.size()[-1] # dimension
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits/torch.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = F.softmax(attn_logits, axis = -1)
    values = torch.matmul(attention, v)
    return values, attention

class AttentionHead(nn.Module):
    def __init__(self, dim_in: int, dim_q: int, dim_k: int):
        super().__init__()
        self.q = nn.Linear(dim_in, dim_q)
        self.k = nn.Linear(dim_in, dim_k)
        self.v = nn.Linear(dim_in, dim_k)

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        return scaled_dot_attention(self.q(query), self.k(key), self.v(value))

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, dim_in: int, dim_q: int, dim_k: int):
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(dim_in, dim_q, dim_k) for _ in range(num_heads)])
        self.o_proj = nn.Linear(num_heads*dim_k, dim_in)

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        value = (
            torch.cat([h(query, key, value) for h in self.heads], dim=-1)
        )
        o = self.o_proj(value)
        return o

# Vectorized Solution, working
# class MultiheadAttention(nn.Module):
#
#     def __init__(self, input_dim, embed_dim, num_heads):
#         super().__init__()
#         assert embed_dim%num_heads == 0, "embedding dimensions should be modulo of num_heads"
#         self.input_dim = input_dim
#         self.embed_dim = embed_dim
#         self.num_heads = num_heads
#         self.head_dim = embed_dim // num_heads
#
#         # Stack all weight matrices 1...h together for efficiency
#         # Note that in many implementations you see "bias=False" which is optional
#         self.qkv_proj = nn.Linear(input_dim, 3*embed_dim)
#         self.o_proj = nn.Linear(embed_dim, embed_dim)
#
#         self._reset_parameters()
#
#     def _reset_parameters(self):
#         # Original Transformer initialization, see PyTorch documentation
#         nn.init.xavier_uniform_(self.qkv_proj.weight)
#         self.qkv_proj.bias.data.fill_(0)
#         nn.init.xavier_uniform_(self.o_proj.weight)
#         self.o_proj.bias.data.fill_(0)
#
#     def forward(self, x, mask=None, return_attention=False):
#         """
#
#         :param x: batch, seq, emb
#         :param mask: batch, seq, emb
#         :param return_attention: Boolean
#         :return: output or output, attention
#         """
#         batch_size, seq_length, _ = x.size()
#         qkv = self.qkv_proj(x) # b, seq, 3*embed_dim
#
#         # get Q, K, V from qkv
#         qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3*self.head_dim)
#         qkv = qkv.permute(0, 2, 1, 3) # [Batch, Head, SeqLen, Dims]
#         q, k, v = qkv.chunk(3, dim=-1)
#
#         values, attention = scaled_dot_attention(q, k, v, mask=mask)
#         values = values.permute(0, 2, 1, 3)  # [Batch, SeqLen, Head, Dims]
#         values = values.reshape(batch_size, seq_length, self.embed_dim)
#         o = self.o_proj(values)
#
#         if return_attention:
#             return o, attention
#         else:
#             return o