import argparse
import os
import sys

import torch
from torch import nn
import torchtext
from torchtext import data
from torchtext import datasets

from eval_args import get_arg_parser
from performance import size_metrics
import utils
from models.components.binarization import (
    Binarize,
)

def main() -> None:
    parser = get_arg_parser()
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() and args.cuda else "cpu"
    print('using device {}'.format(device))


    print('loading datasets...')
    src = data.Field(include_lengths=True,
               init_token='<sos>', eos_token='<eos>', batch_first=True, fix_length=200)
    trg = data.Field(include_lengths=True,
               init_token='<sos>', eos_token='<eos>', batch_first=True)
    
    if args.dataset == 'WMT':
        mt_train = datasets.TranslationDataset(
            path=constants.WMT14_EN_FR_SMALL_TRAIN,
            exts=('.en', '.fr'),
            fields=(src, trg)
        )
        src_vocab, trg_vocab = utils.load_torchtext_wmt_small_vocab()
        src.vocab = src_vocab

        trg.vocab = trg_vocab

        mt_valid = None
    else:
        if args.dataset == 'Multi30k':
            mt_train, mt_valid, mt_test = datasets.Multi30k.splits(
                exts=('.en', '.de'),
                fields=(src, trg),
            )
        elif args.dataset == 'IWSLT':
            mt_train, mt_valid, mt_test = datasets.IWSLT.splits(
                exts=('.en', '.de'),
                fields=(src, trg), 
            )
        else:
            raise Exception("Uknown dataset: {}".format(args.dataset))

        print('loading vocabulary...')

        # mt_dev shares the fields, so it shares their vocab objects
        src.build_vocab(
            mt_train,
            min_freq=args.torchtext_unk,
            max_size=args.torchtext_src_max_vocab,
        )

        trg.build_vocab(
            mt_train,
            max_size=args.torchtext_trg_max_vocab,
        )
        print('loaded vocabulary')
    
    # determine the correct dataset to evaluate
    eval_dataset = mt_train if args.eval_train else mt_valid
    eval_dataset = mt_test if args.eval_test else eval_dataset

    train_loader = data.BucketIterator(
        dataset=eval_dataset,
        batch_size=1,
        sort_key=lambda x: len(x.src), # data.interleave_keys(len(x.src), len(x.trg)),
        sort_within_batch=True,
        device=device
    )

    print('model type: {}'.format(args.model_type))
    model = utils.build_model(parser, src.vocab, trg.vocab)
    if args.load_path is not None:
        model.load_state_dict(torch.load(args.load_path))
    model = model.eval()
    if args.binarize:
        print('binarizing model')
        binarized_model = Binarize(model)
        binarized_model.binarization()
    
    print(model)
    
    model_size = size_metrics.get_model_size(model)
    print("64 bit float: {}".format(size_metrics.get_model_size(model, 64, args.binarize)))
    print("32 bit float: {}".format(size_metrics.get_model_size(model, 32, args.binarize)))
    print("16 bit float: {}".format(size_metrics.get_model_size(model, 16, args.binarize)))

if __name__ == "__main__":
    main()