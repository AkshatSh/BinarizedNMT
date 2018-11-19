import torch
import pickle
import argparse
import os
from tqdm import trange, tqdm
import torch
import torchtext
from torchtext import data
from torchtext import datasets

from train_args import get_arg_parser
import constants


def train():
    src = data.Field()
    trg = data.Field()
    mt_train = datasets.TranslationDataset(
        path=constants.WMT14_EN_FR_VALID, exts=('.en', '.fr'),
        fields=(src, trg),
    )

    src.build_vocab(mt_train, max_size=80000)
    trg.build_vocab(mt_train, max_size=40000)

    train_iter = data.BucketIterator(
        dataset=mt_train,
        batch_size=32,
        sort_key=lambda x: data.interleave_keys(len(x.src), len(x.trg))
    )

    a = (next(iter(train_iter)))
    print(a)
    print('done')

def main(args):
    train()

if __name__ == "__main__":
    parser = get_arg_parser()
    args = parser.parse_args()
    main(args)