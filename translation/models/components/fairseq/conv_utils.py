import math

import torch
from torch import nn

from .learned_positional_embedding import (
    LearnedPositionalEmbedding
)

from .linearized_conv import LinearizedConvolution as fairseq_linear_conv

from .conv_tbc import ConvTBC as fairseq_convtbc

def Embedding(
    num_embeddings: int, 
    embedding_dim: int,
    padding_idx: int,
) -> nn.Module:
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, 0, 0.1)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m

def PositionalEmbedding(
    num_embeddings: int, 
    embedding_dim: int,
    padding_idx: int,
) -> nn.Module:
    # left_pad = False, since everything is right padded here
    m = LearnedPositionalEmbedding(num_embeddings, embedding_dim, padding_idx, False)
    nn.init.normal_(m.weight, 0, 0.1)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m

def Linear(
    in_features: int,
    out_features: int,
    dropout: float=0
) -> nn.Module:
    """Weight-normalized Linear layer (input: N x T x C)"""
    m = nn.Linear(in_features, out_features)
    nn.init.normal_(m.weight, mean=0, std=math.sqrt((1 - dropout) / in_features))
    nn.init.constant_(m.bias, 0)
    return nn.utils.weight_norm(m)

def LinearizedConv1d(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    dropout: float=0,
    **kwargs,
) -> nn.Module:
    """Weight-normalized Conv1d layer optimized for decoding"""
    m = fairseq_linear_conv(in_channels, out_channels, kernel_size, **kwargs)
    std = math.sqrt((4 * (1.0 - dropout)) / (m.kernel_size[0] * in_channels))
    nn.init.normal_(m.weight, mean=0, std=std)
    nn.init.constant_(m.bias, 0)
    return nn.utils.weight_norm(m, dim=2)


def ConvTBC(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    dropout: float=0, 
    **kwargs,
) -> nn.Module:
    """Weight-normalized Conv1d layer"""
    m = fairseq_convtbc(in_channels, out_channels, kernel_size, **kwargs)
    std = math.sqrt((4 * (1.0 - dropout)) / (m.kernel_size[0] * in_channels))
    nn.init.normal_(m.weight, mean=0, std=std)
    nn.init.constant_(m.bias, 0)
    return nn.utils.weight_norm(m, dim=2)