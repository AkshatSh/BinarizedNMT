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
        # context: (batch_size, seq, seq)
        return context

class AttentionModule(nn.Module):
    def __init__(
        self,
        method: str,
        hidden_size: int,
    ):
        super(Attn, self).__init__()

        self.method = method
        self.hidden_size = hidden_size
        self.device = device

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, self.hidden_size)
        elif self.method == 'concat':
            # 2 x for concat
            self.attn = nn.Linear(self.hidden_size * 2, self.hidden_size)
            self.v = nn.Parameter(torch.Tensor(1, hidden_size))
    
    def forward(
        self,
        hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
    ) -> torch.Tensor:
        '''
        Arguments:
            hidden: torch.Tensor of shape (batch, dim, 1)
            encoder_outputs: torch.Tensor of shape (batch, seq_len, dim)
        
        Returns:
            torch.Tensor of next prediction (batch, max_len)
            (encoder_outputs @ hidden) gives the right dimension
        '''
        # (batch, max_len)
        attn_energies = self.score(hidden, encoder_outputs)

        # (batch, max_len)
        return F.softmax(atten_energies, dim=1)
    
    def score(
        self,
        hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
    ) -> torch.Tensor:
        '''
        Arguments:
            hidden: torch.Tensor of shape (batch, dim, 1)
            encoder_outputs: torch.Tensor of shape (batch, seq_len, dim)
        
        Returns:
            torch.Tensor of next prediction (batch, max_len)
            (encoder_outputs @ hidden) gives the right dimension
        '''
        batch_size = encoder_outputs.shape[0]
        seq_len = encoder_outputs.shape[1]
        dim = encoder_outputs.shape[2]
        if self.method == 'dot':
            return encoder_outputs @ hidden
        elif self.method == 'general':
            energy = self.attn(encoder_output)
            return energy @ hidden
        elif self.method == 'concat':
            # hidden = (batch, seq_len, dim)
            hidden = hidden.transpose(1,2).expand(batch_size, seq_len, dim)
            energy = self.attn(
                torch.cat([hidden, encoder_outputs], dim=2)
            )
            return energy @ hidden
        
        # should never be the case
        raise Exception(
            "[Attention]: Not supported method: {}".format(method)
        )
