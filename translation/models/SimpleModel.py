import torch
from torch import nn
import torch.nn.Functional as F

from .EncoderDecoder import EncoderModel
from ..vocab import Vocabulary
from ..constants import UNKOWN_TOKEN

class SimpleLSTMEncoder(EncoderModel):
    def __init__(
        self, 
        input_size: int,
        hidden_size: int,
        en_vocab: Vocabulary,
        fr_vocab: Vocabulary,
    ):
        super(EncoderRNN, self).__init__()
        self.en_vocab = en_vocab
        self.fr_vocab = fr_vocab
        self.embed_tokens = nn.Embedding(
            num_embeddings=len(self.en_vocab),
            embedding_dim=embed_dim,
            padding_idx=self.en_vocab.word2idx(UNKOWN_TOKEN),
        )
        self.dropout = nn.Dropout(p=dropout)

        # We'll use a single-layer, unidirectional LSTM for simplicity.
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            bidirectional=False,
        )

    def forward(
        self,
        input,
        hidden,
    ) -> tuple:
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)