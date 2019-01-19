import os
import mmap
from typing import List
import torch
import numpy as np
import torchtext
import torch
from torch import nn
from torchtext.vocab import Vocab
from typing import Tuple
import pickle
import nltk
import argparse

from nltk.translate.bleu_score import SmoothingFunction
chencherry = SmoothingFunction()

from vocab import Vocabulary
import constants

from models import (
    AttentionRNN,
    AttentionQRNN,
    ConvS2S,
    SimpleLSTMModel,
)

def get_num_lines(file_path: str) -> int:
    '''
    returns the number of lines in a file
    '''
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines

def convert_to_str(
    tensor: np.ndarray,
    vocab: Vocabulary,
) -> List[List[str]]:
    output = []
    for batch in range(len(tensor)):
        curr = []
        for idx in range(len(tensor[batch])):
            curr.append(vocab.idx2word(tensor[batch, idx]))
        output.append(curr)
    return output

def torchtext_convert_to_str(
    tensor: np.ndarray,
    vocab: Vocabulary,
) -> List[List[str]]:
    output = []
    for batch in range(len(tensor)):
        curr = []
        for idx in range(len(tensor[batch])):
            curr.append(vocab.itos[tensor[batch, idx]])
        output.append(curr)
    return output

def load_torchtext_wmt_small_vocab(
) -> Tuple[Vocab, Vocab]:
    '''
    Loads the torchtext wmt small vocabulary files

    Returns:
        Tuple(Vocab, Vocab)
            in the order (src_vocab, trg_vocab)
    '''
    with open(constants.TORCH_TEXT_SMALL_EN_VOCAB_FILE, 'rb') as f:
        src_vocab = pickle.load(f)

    with open(constants.TORCH_TEXT_SMALL_FR_VOCAB_FILE, 'rb') as f:
        trg_vocab = pickle.load(f)
    
    return src_vocab, trg_vocab

def get_raw_sentence(sentence: List[str]) -> List[str]:
    end_index = sentence.index('<eos>') if '<eos>' in sentence else len(sentence)
    start_index = sentence.index('<sos>') if '<sos>' in sentence else -1
    return sentence[start_index + 1:end_index]

def compute_bleu(predicted: List[str], expected: List[str]) -> float:
    '''
    nltk.translate.bleu_score.sentence_bleu([reference], hypothesis)
    '''
    if constants.BLEU_USE_SMOOTHING:
        return nltk.translate.bleu_score.sentence_bleu(
            [expected],
            predicted,
            smoothing_function=chencherry.method4,
        )
    else:
        return nltk.translate.bleu_score.sentence_bleu(
            [expected],
            predicted,
        )

def get_num_parameters(model: nn.Module) -> int:

    params=[
        param for param in model.parameters()
    ]
    # attention RNN: 26,409,236
    # convs2s: 36,301,076
    size = 0
    for param in params:
        curr_size = 1
        for s in param.shape:
            curr_size *= s
        size += curr_size
    return size

def build_model(
    parser: argparse.ArgumentParser,
    en_vocab: Vocabulary,
    fr_vocab: Vocabulary,
) -> nn.Module:
    # TODO make switch case
    args = parser.parse_args()
    if args.model_type == 'SimpleLSTM':
        SimpleLSTMModel.add_args(parser)
        args = parser.parse_args()
        return SimpleLSTMModel.build_model(
            src_vocab=en_vocab,
            trg_vocab=fr_vocab,
            encoder_embed_dim=args.encoder_embed_dim,
            encoder_hidden_dim=args.encoder_hidden_dim,
            encoder_dropout=args.encoder_dropout,
            encoder_num_layers=args.encoder_layers,
            decoder_embed_dim=args.decoder_embed_dim,
            decoder_hidden_dim=args.decoder_hidden_dim,
            decoder_dropout=args.decoder_dropout,
            decoder_num_layers=args.decoder_layers,
        )
    elif args.model_type == 'AttentionRNN':
        AttentionRNN.add_args(parser)
        args = parser.parse_args()
        return AttentionRNN.build_model(
            src_vocab=en_vocab,
            trg_vocab=fr_vocab,
            encoder_embed_dim=args.encoder_embed_dim,
            encoder_hidden_dim=args.encoder_hidden_dim,
            encoder_dropout=args.encoder_dropout,
            encoder_num_layers=args.encoder_layers,
            decoder_embed_dim=args.decoder_embed_dim,
            decoder_hidden_dim=args.decoder_hidden_dim,
            decoder_dropout=args.decoder_dropout,
            decoder_num_layers=args.decoder_layers,
            teacher_student_ratio=args.teacher_student_ratio,
        )
    elif args.model_type == 'AttentionQRNN':
        AttentionQRNN.add_args(parser)
        args = parser.parse_args()
        return AttentionQRNN.build_model(
            src_vocab=en_vocab,
            trg_vocab=fr_vocab,
            encoder_embed_dim=args.encoder_embed_dim,
            encoder_hidden_dim=args.encoder_hidden_dim,
            encoder_dropout=args.encoder_dropout,
            encoder_num_layers=args.encoder_layers,
            decoder_embed_dim=args.decoder_embed_dim,
            decoder_hidden_dim=args.decoder_hidden_dim,
            decoder_dropout=args.decoder_dropout,
            decoder_num_layers=args.decoder_layers,
            teacher_student_ratio=args.teacher_student_ratio,
        )
    elif args.model_type == 'ConvSeq2Seq':
        ConvS2S.add_args(parser)
        args = parser.parse_args()
        return ConvS2S.build_model(
            src_vocab=en_vocab,
            trg_vocab=fr_vocab,
            max_positions=args.max_positions,
            encoder_embed_dim=args.encoder_embed_dim,
            encoder_conv_spec=args.encoder_conv_spec,
            encoder_dropout=args.encoder_dropout,
            decoder_embed_dim=args.decoder_embed_dim,
            decoder_out_embed_dim=args.decoder_out_embed_dim,
            decoder_conv_spec=args.decoder_conv_spec,
            decoder_dropout=args.decoder_dropout,
            decoder_attention=args.decoder_attention,
            share_embed=args.share_embed,
            decoder_positional_embed=args.decoder_positional_embed,
            binarize=args.binarize,
            linear_type=args.linear_type,
        )
    else:
        raise Exception(
            "Unknown Model Type: {}".format(args.model_type)
        )

def create_entry_delim() -> str:
    return "\n<----------------------NEW ENTRY---------------------->"
