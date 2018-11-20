import torch
from torch import nn
from torch
import torch.nn.functional as F
import numpy

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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

class EncoderDecoderModel(nn.Module):
    def __init__(self):
        pass
    
    def forward(
        self,
        prev_output_tokens: torch.Tensor,
        encoder_out: torch.Tensor = None,
    ) -> torch.Tensor:
        pass