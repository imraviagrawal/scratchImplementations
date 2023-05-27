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
from modules import MultiheadAttention

class EncoderBlock(nn.Module):
    def __init__(self, input_dim, num_heads, dim_feedforward, dropout=0.0):
        """
                Inputs:
                    input_dim - Dimensionality of the input
                    num_heads - Number of heads to use in the attention block
                    dim_feedforward - Dimensionality of the hidden layer in the MLP
                    dropout - Dropout probability to use in the dropout layers
                """
        super().__init__()

        # Attention layer
        self.self_attn = MultiheadAttention(input_dim, input_dim, num_heads)
        self.linear_net = nn.Sequential(nn.Linear(input_dim, dim_feedforward),
                                        nn.Dropout(dropout),
                                        nn.ReLU(inplace=True),
                                        nn.Linaer(dim_feedforward, input_dim))
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out = self.self_attn(x, mask=mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        # mlp
        linear_out = self.linear_net(x)
        x = x + self.dropout(linear_out)
        x = self.norm2(x)

        return x

class Transformers(nn.Module):
    def __init__(self, num_layers, **block_args):
        super().__init__()
        self.layers = nn.ModuleList([EncoderBlock(**block_args) for _ in range(num_layers)])

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
