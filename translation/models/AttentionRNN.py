import sys
sys.path.append("..")

import torch
from torch import nn
import torch.nn.functional as F
import random
import argparse

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
        src_vocab: Vocabulary,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ):
        super(EncoderRNN, self).__init__()
        self.input_size = len(src_vocab)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(
            len(src_vocab),
            hidden_size,
        )
        self.lstm = nn.GRU(
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
        # print(embedded.shape)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, src_lengths, batch_first=True)
        outputs, hidden = self.lstm(packed, hidden)
        outputs, outputs_length = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

        # sum up bidirectional outputs to keep hidden size the same
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:]
        # print('output: ', outputs.shape)
        return outputs, hidden

class AttentionDecoderRNN(DecoderModel):
    def __init__(
        self,
        trg_vocab: Vocabulary,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        teacher_student_ratio: float,
    ):
        super(AttentionDecoderRNN, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = len(trg_vocab)
        self.num_layers = num_layers
        self.dropout = dropout
        self.teacher_student_ratio = teacher_student_ratio

        # layers
        self.embedding = nn.Embedding(
            len(trg_vocab),
            hidden_size,
        )

        self.dropout = nn.Dropout(dropout)

        self.attn = AttentionModule('general', hidden_size)

        self.lstm = nn.GRU(
            input_size=hidden_size * 2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=False,
            batch_first=True,
        )

        self.out = nn.Linear(
            hidden_size,
            len(trg_vocab),
        )
    
    def forward(
        self,
        prev_tokens: torch.Tensor,
        encoder_out: tuple,
    ) -> torch.Tensor:
        encoder_outputs, last_hidden = encoder_out
        batch_size, seq_len = prev_tokens.shape
        if random.random() <= self.teacher_student_ratio:
            return self.teacher_forward(
                last_hidden,
                encoder_outputs,
                prev_tokens,
            )
        else:
            return self.student_forward(
                last_hidden,
                encoder_outputs,
                seq_len,
            )

    def teacher_forward(
        self,
        final_hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
        prev_tokens: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len = prev_tokens.shape
        final_hidden = final_hidden[:self.num_layers]
        final_encoder_hidden = final_hidden

        # embedded_prev_tokens: (batch, seq_len, trg_vocab)
        embedded_prev_tokens = self.embedding(prev_tokens)
        embedded_prev_tokens = self.dropout(embedded_prev_tokens)

        decoder_outputs = []
        last_hidden = final_hidden
        
        for i in range(seq_len):
            attn_weights = self.attn(last_hidden[-1], encoder_outputs)

            # encoder_outputs: (batch, seq_len, dim)
            # attn_weights = (batch, seq_len)
            context = attn_weights.transpose(1,2).bmm(encoder_outputs)

            lstm_input = torch.cat((embedded_prev_tokens[:, i:i+1, :], context), dim=2)
            output, last_hidden = self.lstm(lstm_input, last_hidden)
            decoder_outputs.append(output)
        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        out = self.out(decoder_outputs)
        return out, last_hidden            
    
    def student_forward(
        self,
        last_hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        batch_size, seq_len = prev_tokens.shape

def build_model(
    src_vocab: Vocabulary,
    trg_vocab: Vocabulary,
    encoder_embed_dim: int,
    encoder_hidden_dim: int,
    encoder_dropout: float,
    encoder_num_layers: int,
    decoder_embed_dim: int,
    decoder_hidden_dim: int,
    decoder_dropout: float,
    decoder_num_layers: int,
    teacher_student_ratio: float,
) -> nn.Module:
    encoder = EncoderRNN(
        src_vocab=src_vocab,
        hidden_size=encoder_hidden_dim,
        num_layers=encoder_num_layers,
        dropout=encoder_dropout,
    )

    decoder = AttentionDecoderRNN(
        trg_vocab=trg_vocab,
        hidden_size=decoder_hidden_dim,
        num_layers=decoder_num_layers,
        dropout=decoder_dropout,
        teacher_student_ratio=teacher_student_ratio,
    )

    return EncoderDecoderModel(
        encoder,
        decoder,
        src_vocab,
        trg_vocab,
    )

def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--encoder_embed_dim', type=int, default=512, help='Embedding dimension for the encoder')
    parser.add_argument('--encoder_hidden_dim', type=int, default=512, help='The hidden (feature size) for the encoder')
    parser.add_argument('--encoder_dropout', type=float, default=0.2, help='the encoder dropout to apply')
    parser.add_argument('--decoder_embed_dim', type=int, default=512, help='the decoder embedding dimension')
    parser.add_argument('--decoder_hidden_dim', type=int, default=512, help='the hidden (feature size) for the decoder')
    parser.add_argument('--decoder_dropout', type=float, default=0.2, help='the decoder dropout')
    parser.add_argument('--encoder_layers', type=int, default=2, help='the number of layers in the encoder')
    parser.add_argument('--decoder_layers', type=int, default=2, help='the number of layers in the decoder')
    parser.add_argument('--teacher_student_ratio', type=float, default=1.0, help='the ratio of teacher to student to use')