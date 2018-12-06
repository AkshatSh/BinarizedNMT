import sys
sys.path.append("..")

import torch
from torch import nn
import torch.nn.functional as F
import random
import argparse
from torchqrnn import QRNN

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

class EncoderQRNN(EncoderModel):
    def __init__(
        self,
        src_vocab: Vocabulary,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ):
        super(EncoderQRNN, self).__init__()
        self.input_size = len(src_vocab)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(
            len(src_vocab),
            hidden_size,
        )
        self.lstm = QRNN(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
        )
    
    def forward(
        self,
        src_tokens: torch.Tensor,
        src_lengths: torch.Tensor,
        hidden: torch.Tensor = None,
    ) -> torch.Tensor:
        embedded = self.embedding(src_tokens)
        # print(embedded.shape)
        #packed = nn.utils.rnn.pack_padded_sequence(embedded, src_lengths, batch_first=True)
        #packed = packed.t()
        embedded = embedded.transpose(0, 1)
        outputs, hidden = self.lstm(embedded, hidden)
        outputs = outputs.transpose(0, 1)
        #outputs, outputs_length = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

        # sum up bidirectional outputs to keep hidden size the same
        #outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:]
        # print('output: ', outputs.shape)
        return outputs, hidden

class AttentionDecoderQRNN(DecoderModel):
    def __init__(
        self,
        trg_vocab: Vocabulary,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        teacher_student_ratio: float,
    ):
        super(AttentionDecoderQRNN, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = len(trg_vocab)
        self.num_layers = num_layers
        self.dropout = dropout
        self.teacher_student_ratio = teacher_student_ratio
        self.trg_vocab = trg_vocab

        # layers
        self.embedding = nn.Embedding(
            len(trg_vocab),
            hidden_size,
        )

        self.dropout = nn.Dropout(dropout)

        self.attn = AttentionModule('general', hidden_size)

        self.lstm = QRNN(
            input_size=hidden_size * 2,
            hidden_size=hidden_size,
            num_layers=num_layers,
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
    
    def forward_eval(
        self,
        prev_tokens: torch.Tensor,
        encoder_out: tuple,
        intermediate: torch.Tensor,
    ) -> torch.Tensor:
        encoder_outputs, last_hidden = encoder_out
        return self.teacher_forward(
            last_hidden if intermediate is None else intermediate,
            encoder_outputs,
            prev_tokens,
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
            #print(encoder_outputs.shape)

            #print(embedded_prev_tokens.shape, context.shape)
            lstm_input = torch.cat((embedded_prev_tokens[:, i:i+1, :], context), dim=2)
            lstm_input = lstm_input.transpose(0, 1)
            output, last_hidden = self.lstm(lstm_input, last_hidden)
            output = output.transpose(0, 1)
            decoder_outputs.append(output)
        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        out = self.out(decoder_outputs)
        return out, last_hidden            
    
    def student_forward(
        self,
        final_hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        batch_size = encoder_outputs.shape[0]
        final_hidden = final_hidden[:self.num_layers]
        device = final_hidden.device

        prev_output = torch.zeros((batch_size, 1)).long().to(device)
        prev_output[:, 0] = self.trg_vocab.stoi['<sos>']
        final_encoder_hidden = final_hidden

        decoder_outputs = []
        last_hidden = final_hidden
        
        for i in range(seq_len):
            attn_weights = self.attn(last_hidden[-1], encoder_outputs)

            # encoder_outputs: (batch, seq_len, dim)
            # attn_weights = (batch, seq_len)
            context = attn_weights.transpose(1,2).bmm(encoder_outputs)

            embedded_prev_tokens = self.embedding(prev_output)
            embedded_prev_tokens = self.dropout(embedded_prev_tokens)

            lstm_input = torch.cat((embedded_prev_tokens, context), dim=2)
            output, last_hidden = self.lstm(lstm_input, last_hidden)
            output = self.out(output)
            decoder_outputs.append(output)
            topi = output.data.max(2)[1]
            prev_output = topi
        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        return decoder_outputs, last_hidden  

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
    encoder = EncoderQRNN(
        src_vocab=src_vocab,
        hidden_size=encoder_hidden_dim,
        num_layers=encoder_num_layers,
        dropout=encoder_dropout,
    )

    decoder = AttentionDecoderQRNN(
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
