import torch
import pickle
import argparse
import os
from tqdm import trange, tqdm
import torch
import torchtext
from torchtext import data
from torchtext import datasets
from torch import nn

from train_args import get_arg_parser
import constants
from vocab import Vocabulary, load_vocab
import dataset as d

def train(
    train_loader: d.BatchedIterator,
    valid_loader: d.BatchedIterator,
    model: nn.Module,
    learning_rate: float,
    weight_decay: float,
    log_dir: str,
    save_dir: str,
    en_vocab: Vocabulary,
    fr_vocab: Vocabulary,
):
    pass

def main():
    parser = get_arg_parser()
    args = parser.parse_args()

    print('loading vocabulary...')
    en_vocab = load_vocab(constants.TRAIN_EN_VOCAB_FILE)
    fr_vocab = load_vocab(constants.TRAIN_FR_VOCAB_FILE)
    print('loaded vocabulary')

    print('loading datasets...')
    train_dataset = d.ShardedCSVDataset(constants.WMT14_EN_FR_TRAIN_SHARD)
    valid_dataset = d.DualFileDataset(
        constants.WMT14_EN_FR_VALID + ".en",
        constants.WMT14_EN_FR_VALID + ".fr",
    )

    train_loader = d.BatchedIterator(
        args.batch_size,
        train_dataset,
        en_vocab,
        fr_vocab,
        args.max_sequence_length,
    )

    valid_loader = d.BatchedIterator(
        1,
        valid_dataset,
        en_vocab,
        fr_vocab,
        args.max_sequence_length,
    )

    train()

if __name__ == "__main__":
    main()