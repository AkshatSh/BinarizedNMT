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
    Embedding,
    Conv1d,
    Projection,
)

from models.components.binarized_convolution import (
    BinConv1d,
    XNORConv1d,
)

from models.components.binarized_linear import (
    BinLinear,
    XNORLinear,
)

from models.EncoderDecoder import (
    EncoderModel,
    DecoderModel,
    EncoderDecoderModel,
    DecoderOutputType
)

from models.components.attention import (
    AttentionModule,
    AttentionLayer,
)

from vocab import Vocabulary

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
        binarize: bool,
        linear_type: type,
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
        self.padding_idx = src_vocab.stoi['<pad>']

        self.embedding = Embedding(
            len(src_vocab),
            embedding_dim,
            self.padding_idx,
        )

        self.embed_positions = PositionalEmbedding(
            max_positions,
            embedding_dim,
            self.padding_idx,
        )

        in_channels = self.convolution_spec[0][0]

        # convert the embedding dimensions into the input channels
        self.fc1 = linear_type(embedding_dim, in_channels)

        self.projections = nn.ModuleList()
        self.convolutions = nn.ModuleList()
        self.residuals = []

        layer_in_channels = [in_channels]

        for i, (out_channels, kernel_width, residual, conv_type) in enumerate(self.convolution_spec):
            if residual == 0:
                residual_dim = out_channels
            else:
                # connect to the dim of the -residual channel
                residual_dim = layer_in_channels[-residual]
            
            # create a projection to convert the last layer of channels
            # to the current one
            self.projections.append(
                Projection(residual_dim, out_channels)
                if residual_dim != out_channels else None
            )

            if kernel_width % 2 == 1:
                padding = kernel_width // 2
            else:
                padding = 0
            
            if binarize:
                if conv_type == 'xnor':
                    conv_class = XNORConv1d
                elif conv_type == 'bwn':
                    conv_class = BinConv1d
            else:
                conv_class = Conv1d

            # create a convolution for the layer
            self.convolutions.append(
                conv_class(
                    in_channels,
                    out_channels * 2,
                    kernel_width,
                    dropout=dropout,
                    padding=padding,
                )
            )

            # keep track of residual connections for
            # each layer
            self.residuals.append(residual)

            in_channels = out_channels
            layer_in_channels.append(out_channels)
        
        self.fc2 = linear_type(
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
        x = F.dropout(embedded, p=self.dropout, training=self.training)
        input_embedding = x

        # transform input to be ready for convolution
        x =  self.fc1(x)

        # a mask over the padding index
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        # TODO: Investiage if its necessary to transpose to T x B x C
        # currently going to assume it is not and go forward
        # B x T x C -> B x C x T
        x = x.transpose(1, 2)

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
                x = x.masked_fill(encoder_padding_mask.unsqueeze(1), 0)
            
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
            x = F.glu(x, dim=1)
            # x = F.relu(x[:, :x.shape[1] // 2, :] + x[:,x.shape[1] // 2:, :])

            if residual is not None:
                # connect with residual layer
                # TODO why $$\sqrt{\frac{1}{2}}$$
                x = (x + residual) * math.sqrt(0.5)
            
            residuals.append(x)
        
        # B x C x T -> B x T x C
        x = x.transpose(1, 2)

        x = self.fc2(x)
        if encoder_padding_mask is not None:
            # encoder_padding_mask = encoder_padding_mask.transpose(1,2)
            x = x.masked_fill(encoder_padding_mask.unsqueeze(-1), 0)

        
        # add output to input embedding for attention
        y = (x + input_embedding) * math.sqrt(0.5)

        return (
            (x, y), # encoder outputs
            encoder_padding_mask, # padding mask
        )

class ConvDecoder(DecoderModel):
    def __init__(
        self,
        trg_dictionary: Vocab,
        embed_dim: int,
        out_embed_dim: int,
        max_positions: int,
        convolution_spec: ConvSpecType,
        attention: bool,
        dropout: float,
        share_embed: bool,
        positional_embedding: bool,
        binarize: bool,
        linear_type: type,
    ):
        super(ConvDecoder, self).__init__()
        self.trg_dictionary = trg_dictionary
        self.embed_dim = embed_dim
        self.out_embed_dim = out_embed_dim
        self.max_positions = max_positions
        self.convolution_spec = convolution_spec
        self.need_attn = True

        if isinstance(attention, bool):
            self.attention = [attention] * len(convolution_spec)
        elif isinstance(attention, list):
            self.attention = attention
        else:
            raise Exception("Unexpected type for attention: {}".format(attention))

        self.dropout = dropout
        self.share_embed = share_embed
        self.positional_embedding = positional_embedding

        num_embeddings = len(trg_dictionary)
        padding_idx = trg_dictionary.stoi['<pad>']
        self.embed_tokens = Embedding(
            num_embeddings,
            embed_dim,
            padding_idx,
        )

        self.embed_positions = PositionalEmbedding(
            max_positions,
            embed_dim,
            padding_idx,
        ) if positional_embedding else None

        in_channels = self.convolution_spec[0][0]



        self.fc1 = linear_type(embed_dim, in_channels, dropout=dropout)
        self.projections = nn.ModuleList()
        self.convolutions = nn.ModuleList()
        self.attentions = nn.ModuleList()
        self.residuals = []
        layer_in_channels = [in_channels]
        for i, (out_channels, kernel_width, residual, conv_type) in enumerate(self.convolution_spec):
            if residual == 0:
                residual_dim = out_channels
            else:
                # connect to the dim of the -residual channel
                residual_dim = layer_in_channels[-residual]
            
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
            
            if binarize:
                if conv_type == 'xnor':
                    conv_class = XNORConv1d
                elif conv_type == 'bwn':
                    conv_class = BinConv1d
            else:
                conv_class = Conv1d
            
            # create a convolution for the layer
            self.convolutions.append(
                conv_class(
                    in_channels,
                    out_channels * 2,
                    kernel_width,
                    padding=(kernel_width - 1),
                    dropout=dropout,
                )
            )

            # keep track of residual connections for
            # each layer
            self.residuals.append(residual)

            # add Attention
            self.attentions.append(
                AttentionLayer(
                    out_channels,
                    embed_dim,
                    linear_type,
                ) if self.attention[i] else None
            )

            in_channels = out_channels
            layer_in_channels.append(out_channels)

        self.fc2 = linear_type(
            out_channels,
            out_embed_dim,
        )

        self.fc3 = linear_type(
            out_embed_dim,
            num_embeddings,
        )
    
    def forward(
        self,
        prev_tokens: torch.Tensor,
        encoder_out: tuple,
        intermediate_state: dict = None,
    ) -> torch.Tensor:
        (encoder_a, encoder_b), encoder_padding_mask = encoder_out
        encoder_a = encoder_a.transpose(1, 2).contiguous()

        pos_embed = 0
        if self.embed_positions is not None:
            pos_embed = self.embed_positions(
                prev_tokens,
            )
        
        if intermediate_state is not None:
            prev_tokens = prev_tokens[:, -1:]
        x = self.embed_tokens(prev_tokens)

        # add the positional embedding
        x += pos_embed
        x = F.dropout(x, p=self.dropout, training=self.training)
        target_embedding = x

        # start convolutional layers
        x = self.fc1(x)

        num_attn_layers = len(self.attentions)
        residuals = [x]

        for i, (proj, conv, attn, res_layer) in \
            enumerate(
                zip(self.projections, self.convolutions, self.attentions, self.residuals)
            ):

            residual = None
            if res_layer > 0:
                # residual exists
                residual = residuals[-res_layer]
                residual = proj(residual) if proj is not None else residual

            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x.transpose(1, 2)).transpose(2, 1)
            if conv.padding[0] > 0:
                x = x[:, :-conv.padding[0], :]  # remove future timestamps
            x = F.glu(x, dim=2)
            # x = F.relu(x[:, :, :x.shape[2] // 2] + x[:, :, x.shape[2] // 2:])

            # attention
            if attn is not None:
                x, attn_scores = attn(x, target_embedding, (encoder_a, encoder_b), encoder_padding_mask)
            
            # residual connection
            if residual is not None:
                x = (x + residual) * math.sqrt(0.5)
            residuals.append(x)

        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc3(x)
        x = F.log_softmax(x, dim=2)

        return x, None

def build_model(
    src_vocab: Vocabulary,
    trg_vocab: Vocabulary,
    max_positions: int,
    encoder_embed_dim: int,
    encoder_conv_spec: ConvSpecType,
    encoder_dropout: float,
    decoder_embed_dim: int,
    decoder_out_embed_dim: int,
    decoder_conv_spec: ConvSpecType,
    decoder_dropout: float,
    decoder_attention: bool,
    share_embed: bool,
    decoder_positional_embed: bool,
    binarize: bool,
    linear_type: str,
) -> nn.Module:
    if binarize:
        if linear_type == 'bwn':
            linear_class = BinLinear
        elif linear_type == 'xnor':
            linear_class = XNORLinear
    else:
        linear_class = Linear
    encoder = ConvEncoder(
        src_vocab=src_vocab,
        embedding_dim=encoder_embed_dim,
        max_positions=max_positions,
        convolution_spec=encoder_conv_spec,
        dropout=encoder_dropout,
        binarize=binarize,
        linear_type=linear_class,
    )

    decoder = ConvDecoder(
        trg_dictionary=trg_vocab,
        embed_dim=decoder_embed_dim,
        out_embed_dim=decoder_out_embed_dim,
        max_positions=max_positions,
        convolution_spec=decoder_conv_spec,
        attention=decoder_attention,
        dropout=decoder_dropout,
        share_embed=share_embed,
        positional_embedding=decoder_positional_embed,
        binarize=binarize,
        linear_type=linear_class,
    )

    encoder.num_attention_layers = sum(layer is not None for layer in decoder.attentions)

    return EncoderDecoderModel(
        encoder,
        decoder,
        src_vocab,
        trg_vocab,
    )

def get_default_conv_spec() -> ConvSpecType:
    # convs = '[(512, 3, 1)] * 9'  # first 9 layers have 512 units
    # convs += ' + [(1024, 3, 1)] * 4'  # next 4 layers have 1024 units
    # convs += ' + [(2048, 1, 1)] * 2'  # final 2 layers use 1x1 convolutions
    # Above architecture experiences exploding gradients
    
    convs = "[(256, 3, 1, 'xnor')] * 4"
    bin_conv = '[(512, 3, 1)] * 4'
    # convs = '[(256, 3, 1)] * 2 + [(512, 3, 1)] * 2'

    # convs = '[(512, 3, 1)] * 2 + [(1024, 3, 1)] * 2' # + [(2048, 1, 1)] * 2'

    # convs = '[(512,3,1)] * 3 + [(1024, 3, 1)] * 4'
    # above architecture experiences exploding gradients
    return eval(convs)

def default_args(parser: argparse.ArgumentParser) -> None:
    # model hyper parameters
    parser.add_argument('--max_positions', type=int, default=1024, help='the maximum positions for hte positional embedding')

    # encoder hyper parameters
    parser.add_argument('--encoder_embed_dim', type=int, default=768, help='the embedding dimension for the encoder')
    parser.add_argument(
        '--encoder_conv_spec', 
        type=ConvSpecType,
        default=get_default_conv_spec(),
        help='convolutional spec for the encoder',
    )
    parser.add_argument('--encoder_dropout', type=float, default=0.1, help='dropout for the encoder')

    # decoder hyper parameters
    parser.add_argument('--decoder_embed_dim', type=int, default=768, help='the decoder embedding dimension')
    parser.add_argument(
        '--decoder_conv_spec', 
        type=ConvSpecType,
        default=get_default_conv_spec(),
        help='convolutional spec for the encoder',
    )
    parser.add_argument('--decoder_out_embed_dim', type=int, default=512, help='the output embedding dimension')
    parser.add_argument('--decoder_dropout', type=float, default=0.1, help='dropout for the decoder')
    parser.add_argument('--decoder_attention', type=bool, default=True, help='whether to use attention for the decoder')
    parser.add_argument('--share_embed', type=bool, default=False, help='whether to share the embedding layer')
    parser.add_argument('--decoder_positional_embed', type=bool, default=True, help='whether to use the positional embeddings')

def multi30k_args(parser: argparse.ArgumentParser) -> None:
    # model hyper parameters
    parser.add_argument('--max_positions', type=int, default=1024, help='the maximum positions for the positional embedding')

    # encoder hyper parameters
    parser.add_argument('--encoder_embed_dim', type=int, default=256, help='the embedding dimension for the encoder')
    parser.add_argument(
        '--encoder_conv_spec', 
        type=ConvSpecType,
        default=get_default_conv_spec(),
        help='convolutional spec for the encoder',
    )
    parser.add_argument('--encoder_dropout', type=float, default=0.1, help='dropout for the encoder')

    # decoder hyper parameters
    parser.add_argument('--decoder_embed_dim', type=int, default=256, help='the decoder embedding dimension')
    parser.add_argument(
        '--decoder_conv_spec', 
        type=ConvSpecType,
        default=get_default_conv_spec(),
        help='convolutional spec for the encoder',
    )
    parser.add_argument('--decoder_out_embed_dim', type=int, default=256, help='the output embedding dimension')
    parser.add_argument('--decoder_dropout', type=float, default=0.1, help='dropout for the decoder')
    parser.add_argument('--decoder_attention', type=bool, default=True, help='whether to use attention for the decoder')
    parser.add_argument('--share_embed', type=bool, default=False, help='whether to share the embedding layer')
    parser.add_argument('--decoder_positional_embed', type=bool, default=True, help='whether to use the positional embeddings')
    parser.add_argument('--linear_type', type=str, default='xnor', help='the type of linear layer to use (None, bwn, xnor)')

def add_args(parser: argparse.ArgumentParser) -> None:
    multi30k_args(parser)
    # default_args(parser)