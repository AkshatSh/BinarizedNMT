import torch
from torch import nn
import numpy
import tqdm
from typing import List

from models.EncoderDecoder import EncoderDecoderModel
from vocab import Vocabulary
from constants import (
    START_TOKEN,
    END_TOKEN,
)

BatchedOutputType = List[str]

def generate_max(
    model: EncoderDecoderModel,
    max_seq_len: int,
    batch_src: torch.Tensor,
    batch_src_length: torch.Tensor,
    en_vocab: Vocabulary,
    fr_vocab: Vocabulary,
) -> BatchedOutputType:
    batch_size = batch_src.shape[0]
    output = torch.zeros((batch_size, max_seq_len))
    encoder_out = model.encoder(batch_src, batch_src_length)
    output[:,:1] = fr_vocab.word2idx(START_TOKEN)
    for i in range(max_seq_len):
        model.decoder()



def generate_sample(
    model: EncoderDecoderModel,
    max_seq_len: int,
    batch_src: str,
    batch_src_length: int,
) -> BatchedOutputType:
    pass

def generate_beam(
    model: EncoderDecoderModel,
    max_seq_len: int,
    batch_src: str,
    batch_src_length: int,
    beam: int,
) -> BatchedOutputType:
    pass
