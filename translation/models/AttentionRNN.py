import sys
sys.path.append("..")

import torch
from torch import nn
import torch.nn.functional as F

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
        pass