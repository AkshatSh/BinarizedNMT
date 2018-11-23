import torch
from torch import nn
from torch
import torch.nn.functional as F
import numpy

from ..vocab import Vocabulary
from ..constants import UNKNOWN_TOKEN

'''
Abstract classes for Encoder Decoder models to be used as the base for
other models
'''

class EncoderModel(nn.Module):
    def __init__(self):
        pass
    
    def forward(
        self,
        src_tokens: torch.Tensor,
        src_lengths: torch.Tensor,
    ) -> torch.Tensor:
        pass

class DecoderModel(nn.Module):
    def __init__(self):
        pass
    
    def forward(
        self,
        prev_output_tokens: torch.Tensor,
        encoder_out: dict,
    ) -> tuple:
        pass

class EncoderDecoderModel(nn.Module):
    def __init__(
        self, 
        encoder: EncoderModel,
        decoder: DecoderModel,
        en_vocab: Vocabulary,
        fr_vocab: Vocabulary,
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.en_vocab = en_vocab
        self.fr_vocab = fr_vocab

    def forward(
        self,
        src_tokens: torch.Tensor,
        src_lengths: torch.Tensor,
        prev_output_tokens: torch.Tensor,
    ) -> torch.Tensor:
        encoder_out = self.encoder(src_tokens, src_lengths)
        decoder_out = self.decoder(prev_output_tokens, encoder_out)
        return decoder_out

    def loss(
        self,
        predicted: torch.Tensor,
        expected: torch.Tensor,
    ):
        return F.cross_entropy(
            predicted,
            expected,
            ignore_index=self.fr_vocab.word2idx(UNKNOWN_TOKEN),
        )