import sys
sys.path.append("..")

import torch
from torch import nn
import torch.nn.functional as F
import random

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

class EncoderRNN(EncoderModel):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(
            input_size,
            hidden_size,
        )
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
        )
    
    def forward(
        self,
        src_tokens: torch.Tensor,
        src_lengths: torch.Tensor,
        hidden: torch.Tensor = None,
    ) -> torch.Tensor:
        embedded = self.embedding(src_tokens)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, src_lengths)
        outputs, hidden = self.lstm(packed, hidden)
        outputs, outputs_length = nn.utils.rnn.pad_packed_sequence(outputs)

        # sum up bidirectional outputs to keep hidden size the same
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:]
        return outputs, hidden

class AttentionDecoderRNN(DecoderModel):
    def __init__(
        self,
        hidden_size: int,
        output_size: int,
        num_layers: int,
        dropout: float,
        teacher_student_ratio: float,
    ):
        super(AttentionDecoderRNN, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.teacher_student_ratio = teacher_student_ratio

        # layers
        self.embedding = nn.Embedding(
            output_size,
            hidden_size,
        )

        self.dropout = nn.Dropout(dropout)

        self.attn = AttentionModule('concat', hidden_size)

        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=False,
            batch_first=True,
        )

        self.out = nn.Linear(
            hidden_size,
            output_size,
        )
    
    def forward(
        self,
        last_hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
        prev_tokens: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len = prev_tokens.shape
        if random.random() < self.teacher_student_ratio:
            return self.teacher_forward(
                self,
                last_hidden,
                encoder_outputs,
                prev_tokens,
            )
        else:
            return self.student_forward(
                self,
                last_hidden,
                encoder_outputs,
                seq_len,
            )

    def teacher_forward(
        self,
        last_hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
        prev_tokens: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len = prev_tokens.shape

        # embedded_prev_tokens: (batch, seq_len, trg_vocab)
        embedded_prev_tokens = self.embedding(prev_tokens)
        embedded_prev_tokens = self.dropout(embedded_prev_tokens)

        decoder_outputs = []
        for i in range(seq_len):
            attn_weights = self.attn(last_hidden[-1], encoder_outputs)

            # encoder_outputs: (batch, seq_len, dim)
            # attn_weights = (batch, seq_len)
            context = attn_weights.bmm(encoder_outputs)

            # TODO: encoder_outputs.transpose(1,2) @ attn_weights.unsqueeze(2)
            lstm_input = torch.cat((embedded_prev_tokens[:, i, :], context), dim=2)
            output, last_hidden = self.lstm(lstm_input, last_hidden)
            decoder_outputs.append(output)
        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        out = self.out(decoder_outputs)
        out = F.softmax(out)
        return out, last_hidden            
    
    def student_forward(
        self,
        last_hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        batch_size, seq_len = prev_tokens.shape