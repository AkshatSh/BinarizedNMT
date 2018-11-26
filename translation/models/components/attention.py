import torch
from torch import nn
import torch.nn.functional as F

class GlobalLuongAttention(nn.Module):
    def __init__(
        self,
        attention_dim: int,
    ):
        '''
        Luong Attention from Effective Approaches to Attention-based Neural Machine Translation
        https://arxiv.org/pdf/1508.04025.pdf

        Thanks to A-Jacobson:
        https://github.com/A-Jacobson/minimal-nmt/blob/master/attention.py
        '''
        super(Attention, self).__init__()
        self.lin_in = nn.Linear(attention_dim, attention_dim)
        self.softmax = nn.Softmax(dim=-1)
    
    def score(
        self,
        encoder_outs: torch.Tensor,
        decoder_hidden: torch.Tensor,
    ) -> torch.Tensor:
        # W^T * x 
        # Encoder_outs shape:
        #       (batch_size, seq_len, dim)
        # Decoder_hidden shape:
        #       (batch_size, 1, dim)
        x = self.lin_in(encoder_output)
        # x is now (batch_size, seq_len, dim)

        # returns shape: (batch_size, seq_len, dim)
        return torch.dot(
            x,
            decoder_hidden.transpose(1,2),
        )
        
    
    def forward(
        self,
        decoder_hidden: torch.Tensor,
        encoder_outs: torch.Tensor,
    ) -> torch.Tensor:
        '''
        Compute a weighted sum based on attn_conditional, and multiply input_tensor
        by the weighted sum

        encoder_outs: (batch, seq, dim)
        decoder_outs: (batch, 1, dim)
        '''
        score = self.score(encoder_outs, decoder_hidden)

        # p_score: (batch_size, sequence_len, dim)
        p_score = self.softmax(score)

        # context: (batch, seq, dim) * (batch, dim, sequence)
        context = p_score @ encoder_outs.transpose(1,2)
        # context: (batch_size, dim, 1)
        return context
