"""
Scratch Implementation of Transformers,

Main Blocks are

Embedding + Position Embedding

LayerNorm(x + MultiAttention(x))

Linaer = Max(0, XW1+b1)W2+b2
LayerNorm(x + Linear(x))

Linear Layer
Softmax

If you follow this file you will learn how transformers works, and implement transformer and
its building blocks from scratch using
pytorch.
"""
import torch
from torch import nn
from modules import MultiHeadAttention

class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        dim_model: int = 512,
        num_heads: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        dim_q = dim_k = max(dim_model // num_heads, 1)

        # Attention layer
        self.self_attn = MultiHeadAttention(num_heads, dim_model, dim_q, dim_k)
        self.linear_net = nn.Sequential(nn.Linear(dim_model, dim_feedforward),
                                        nn.Dropout(dropout),
                                        nn.ReLU(inplace=True),
                                        nn.Linaer(dim_feedforward, dim_model))
        self.norm1 = nn.LayerNorm(dim_model)
        self.norm2 = nn.LayerNorm(dim_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, mask=None):
        attn_out = self.self_attn(src, src, src)
        x = src + self.dropout(attn_out)
        x = self.norm1(x)

        # mlp
        linear_out = self.linear_net(x)
        x = x + self.dropout(linear_out)
        x = self.norm2(x)

        return x

class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, **block_args):
        super().__init__()
        self.layers = nn.ModuleList([TransformerEncoderLayer(**block_args) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        for l in self.layers:
            x = l(x, mask=mask)
        return x

    def get_attention_maps(self, x, mask=None):
        attention_maps = []
        for l in self.layers:
            _, attn_map = l.self_attn(x, mask=mask, return_attention=True)
            attention_maps.append(attn_map)
            x = l(x)
        return attention_maps

class TransformerDecoderLayer(nn.Module):
    def __init__(self, dim_model: int = 512,
        num_heads: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        dim_q = dim_k = max(dim_model // num_heads, 1)
        self.attn1 = MultiHeadAttention(num_heads, dim_model, dim_q, dim_q)
        self.attn2 = MultiHeadAttention(num_heads, dim_model, dim_q, dim_k)

        self.linear_net = nn.Sequential(nn.Linear(dim_model, dim_feedforward),
                                        nn.Dropout(dropout),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(dim_feedforward, dim_model)
                                        )
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(dim_model)
        self.norm2 = nn.LayerNorm(dim_model)
        self.norm3 = nn.LayerNorm(dim_model)

    def forward(self, src, tgt):
        attn1 = self.attn1(src, src, src)
        x = src + self.dropout(attn1)
        x = self.norm1(x)

        attn2 = self.attn2(src, tgt, tgt)
        x = src + self.dropout(attn2)
        x = self.norm2(x)

        x = self.linear_net(x)
        x = src + self.dropout(x)
        x = self.norm3(x)
        return x

class TransformerDecoder(nn.Module):
    def __init__(self,
                 num_layers: int = 32,
                 dim_model = 512,
                 num_heads = 12,
                 dim_feedforward = 2048,
                 dropout: float = 0.1,
                 ):
        self.layers = nn.ModuleList([TransformerDecoderLayer(dim_model, num_heads, dim_feedforward, dropout) for _ in range(num_layers)])
        self.linear = nn.Linear(dim_model, dim_feedforward)

    def forward(self, src, tgt):
        seq_len, dimension = src.size(1), src.size(2)
        for layer in self.layers:
            src = layer(src, tgt)
        return torch.softmax(self.linear(src), dim=-1)



