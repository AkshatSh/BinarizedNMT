import sys
sys.path.append("..")

import math
import random
import argparse
import torch
from typing import (
    Tuple,
    List,
)

from torch import nn
from torchtext.vocab import Vocab
import torch.nn.functional as F

# FAIRseq dependencies
from models.components.fairseq.conv_utils import (
    Linear,
    PositionalEmbedding,
    ConvTBC,
    Embedding,
)

from models.EncoderDecoder import (
    EncoderModel,
    DecoderModel,
    EncoderDecoderModel,
    DecoderOutputType,
)

from models.components.attention import (
    AttentionModule
)

from vocab import Vocabulary

from constants import (
    UNKNOWN_TOKEN,
    PAD_TOKEN,
)

'''
Implementation of the convolutional sequence to sequence architecture described here:
https://arxiv.org/pdf/1705.03122.pdf 

Adopted from: Fair Seq https://github.com/pytorch/fairseq/blob/master/fairseq/models/fconv.py

TODO: there is lots of logic for even sized kernels, but the hyper parameter tuning shows
they aren't even used for translation? It will be simpler if this code is cut.
'''

ConvSpecEntry = Tuple[
    int, # out_channels
    int, # kernel width
    int, # residual connection (how many layers back to connect)
]

ConvSpecType = List[
    ConvSpecEntry
] 

class ConvEncoder(EncoderModel):
    def __init__(
        self,
        src_vocab: Vocab,
        embedding_dim: int,
        max_positions: int,
        convolution_spec: ConvSpecType,
        dropout: float,
    ):
        '''
        Arguments:
            src_vocab: the torchtext vocab object for the src language
            embedding_dim: the dimensions for the embedding
            max_positions: the maximum positions to use
            convolution_spec: 
        '''
        super(ConvEncoder, self).__init__()
        self.src_vocab = src_vocab
        self.max_positions = max_positions
        self.convolution_spec = convolution_spec
        self.dropout = dropout
        self.padding_idx = src_vocab['<pad>']

        self.embedding = Embedding(
            len(src_vocab),
            embedding_dim,
        )

        self.embed_positions = PositionalEmbedding(
            max_positions,
            embed_dim,
            self.padding_idx,
        )

        in_channels = self.convolution_spec[0][0]

        # convert the embedding dimensions into the input channels
        self.fc1 = Linear(embedding_dim, in_channels)

        self.projections = nn.ModuleList()
        self.convolutions = nn.ModuleList()
        self.residuals = []

        layer_in_channels = [in_channels]

        for i, (out_channels, kernel_width, residual) in enumerate(self.convolution_spec):
            if residual == 0:
                residual_dim = out_channels
            else:
                # connect to the dim of the -residual channel
                resitudal_dim = layer_in_channels[-residual]
            
            # create a projection to convert the last layer of channels
            # to the current one
            self.projections.append(
                Linear(residual_dim, out_channels)
                if residual_dim != out_channels else None
            )

            if kernel_width % 2 == 1:
                padding = kernel_width // 2
            else:
                padding = 0
            
            # create a convolution for the layer
            self.convolutions.append(
                ConvTBC(
                    in_channels,
                    out_channels * 2,
                    kernel_size,
                    dropout=dropout,
                    padding=padding,
                )
            )

            # keep track of residual connections for
            # each layer
            self.residuals.append(residual)

            in_channels = out_channels
            layer_in_channels.append(out_channels)
        
        self.fc2 = Linear(
            in_channels,
            embedding_dim,
        )
    
    def forward(
        self,
        src_tokens: torch.Tensor,
        src_lengths: torch.Tensor,
    ) -> torch.Tensor:
        # embed_tokens and positions
        embedded = self.embedding(src_tokens) + self.embed_positions(src_tokens)
        x = F.dropout(x, p=self.dropout, training=self.training)
        input_embedding = x

        # a mask over the padding index
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        # TODO: Investiage if its necessary to transpose to T x B x C
        # currently going to assume it is not and go forward

        # residuals stores the results from each layer, for future residual
        # connections
        residuals = [x]

        for i, (proj, conv, res_layer) in enumerate(
                zip(self.projections, self.convolutions, self.residuals)
            ):

            # default no residual connection
            residual = None
            if res_layer > 0:
                # there is a residual connection
                residual = residuals[-res_layer]
                residual = proj(residual) if proj else residual
            
            # zero out all the padded indexes using the precomputed mask
            if encoder_padding_mask is not None:
                x = x.masked_fill(encoder_padding_mask.unsqueeze(-1), 0)
            
            # TODO: is this dropout necessary? can the both be applied
            # at the same time
            x = F.dropout(x, p=self.dropout, training=self.training)

            if conv.kernel_size[0] % 2 == 1:
                # no padding necessary for odd length kernels
                x = conv(x)
            else:
                # pad both sides
                padding_l = (conv.kernel_size[0] - 1) // 2
                padding_r = conv.kernel_size[0] // 2

                # TODO: what does F.pad do?
                x = F.pad(x, (0, 0, 0, 0, padding_l, padding_r))
                x = conv(x)
            
            # Apply a gated linear unit after each layer
            x = F.glu(x, dim=2)

            if residual is not None:
                # connect with residual layer
                # TODO why $$\sqrt{\frac{1}{2}}$$
                x = (x + residual) * math.sqrt(0.5)

        x = self.fc2(x)
        if encoder_padding_mask is not None:
            encoder_padding_mask = encoder_padding_mask
            x = x.masked_fill(encoder_padding_mask.unsqueeze(-1), 0)
        
        # TODO: why????
        # scale gradients (this only affects backward, not forward)
        x = GradMultiply.apply(x, 1.0 / (2.0 * self.num_attention_layers))

        
        # add output to input embedding for attention
        y = (x + input_embedding) * math.sqrt(0.5)

        return (
            (x, y), # encoder outputs
            encoder_padding_mask, # padding mask
        )

