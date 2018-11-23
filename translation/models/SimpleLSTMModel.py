import sys
sys.path.append("..")

import torch
from torch import nn
import torch.nn.functional as F
import argparse

from models.EncoderDecoder import (
    EncoderModel,
    DecoderModel,
    EncoderDecoderModel,
    DecoderOutputType,
)

from vocab import Vocabulary

from constants import (
    UNKNOWN_TOKEN,
    PAD_TOKEN,
)

'''
Adopted from FAIR Seq Tutorial:

https://fairseq.readthedocs.io/en/latest/tutorial_simple_lstm.html
'''

class SimpleLSTMEncoder(EncoderModel):
    def __init__(
        self,
        embed_dim: int,
        hidden_size: int,
        dropout: float,
        en_vocab: Vocabulary,
        fr_vocab: Vocabulary,
    ):
        super(SimpleLSTMEncoder, self).__init__()
        self.en_vocab = en_vocab
        self.fr_vocab = fr_vocab
        self.embed_tokens = nn.Embedding(
            num_embeddings=len(self.en_vocab),
            embedding_dim=embed_dim,
            padding_idx=self.en_vocab.word2idx(PAD_TOKEN),
        )
        self.dropout = nn.Dropout(p=dropout)

        # We'll use a single-layer, unidirectional LSTM for simplicity.
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=1,
            bidirectional=False,
        )

    def forward(
        self,
        src_tokens: torch.Tensor,
        src_lengths: torch.Tensor,
    ) -> torch.Tensor:
        # Embed the source.
        x = self.embed_tokens(src_tokens)

        # Apply dropout.
        x = self.dropout(x)

        # Pack the sequence into a PackedSequence object to feed to the LSTM.
        x = nn.utils.rnn.pack_padded_sequence(x, src_lengths, batch_first=True)

        # Get the output from the LSTM.
        self.lstm.flatten_parameters()
        _outputs, (final_hidden, _final_cell) = self.lstm(x)

        # Return the Encoder's output. This can be any object and will be
        # passed directly to the Decoder.
        # this will have shape `(bsz, hidden_dim)`
        return final_hidden.squeeze(0)

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

class SimpleLSTMDecoder(DecoderModel):

    def __init__(
        self,
        encoder_hidden_dim: int,
        embed_dim: int,
        hidden_dim: int,
        dropout: float,
        en_vocab: Vocabulary,
        fr_vocab: Vocabulary,
    ):
        super(SimpleLSTMDecoder, self).__init__()

        self.en_vocab = en_vocab
        self.fr_vocab = fr_vocab

        # Our decoder will embed the inputs before feeding them to the LSTM.
        self.embed_tokens = nn.Embedding(
            num_embeddings=len(fr_vocab),
            embedding_dim=embed_dim,
            padding_idx=fr_vocab.word2idx(PAD_TOKEN),
        )
        self.dropout = nn.Dropout(p=dropout)

        # We'll use a single-layer, unidirectional LSTM for simplicity.
        self.lstm = nn.LSTM(
            # For the first layer we'll concatenate the Encoder's final hidden
            # state with the embedded target tokens.
            input_size=encoder_hidden_dim + embed_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            bidirectional=False,
            batch_first=True,
        )

        # Define the output projection.
        self.output_projection = nn.Linear(hidden_dim, len(fr_vocab))
    
    def forward(
        self,
        prev_output_tokens: torch.Tensor,
        encoder_out: torch.Tensor,
    ) -> DecoderOutputType:
        bsz, tgt_len = prev_output_tokens.size()

        # Extract the final hidden state from the Encoder.
        final_encoder_hidden = encoder_out

        # Embed the target sequence, which has been shifted right by one
        # position and now starts with the end-of-sentence symbol.
        x = self.embed_tokens(prev_output_tokens)

        # Apply dropout.
        x = self.dropout(x)

        # Concatenate the Encoder's final hidden state to *every* embedded
        # target token.
        x = torch.cat(
            [x, final_encoder_hidden.unsqueeze(1).expand(bsz, tgt_len, -1)],
            dim=2,
        )

        # Using PackedSequence objects in the Decoder is harder than in the
        # Encoder, since the targets are not sorted in descending length order,
        # which is a requirement of ``pack_padded_sequence()``. Instead we'll
        # feed nn.LSTM directly.
        initial_state = (
            final_encoder_hidden.unsqueeze(0),  # hidden
            torch.zeros_like(final_encoder_hidden).unsqueeze(0),  # cell
        )

        self.lstm.flatten_parameters()
        x, _ = self.lstm(
            x,
            initial_state,
        )

        # Project the outputs to the size of the vocabulary.
        x = self.output_projection(x)

        # Return the logits and ``None`` for the attention weights
        return x, None
    

def build_model(
    en_vocab: Vocabulary,
    fr_vocab: Vocabulary,
    encoder_embed_dim: int,
    encoder_hidden_dim: int,
    encoder_dropout: float,
    decoder_embed_dim: int,
    decoder_hidden_dim: int,
    decoder_dropout: float,
):
    encoder = SimpleLSTMEncoder(
        embed_dim=encoder_embed_dim,
        hidden_size=encoder_hidden_dim,
        dropout=encoder_dropout,
        en_vocab=en_vocab,
        fr_vocab=fr_vocab,
    )

    decoder = SimpleLSTMDecoder(
        encoder_hidden_dim=encoder_hidden_dim,
        dropout=decoder_dropout,
        embed_dim=decoder_embed_dim,
        hidden_dim=decoder_hidden_dim,
        en_vocab=en_vocab,
        fr_vocab=fr_vocab,
    )

    return EncoderDecoderModel(
        encoder,
        decoder,
        en_vocab,
        fr_vocab,
    )

def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--encoder_embed_dim', type=int, default=256, help='Embedding dimension for the encoder')
    parser.add_argument('--encoder_hidden_dim', type=int, default=256, help='The hidden (feature size) for the encoder')
    parser.add_argument('--encoder_dropout', type=float, default=0.2, help='the encoder dropout to apply')
    parser.add_argument('--decoder_embed_dim', type=int, default=256, help='the decoder embedding dimension')
    parser.add_argument('--decoder_hidden_dim', type=int, default=256, help='the hidden (feature size) for the decoder')
    parser.add_argument('--decoder_dropout', type=float, default=0.2, help='the decoder dropout')