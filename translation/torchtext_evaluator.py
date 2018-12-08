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
import torch.nn.functional as F
import math

from models import SimpleLSTMModel, AttentionRNN

from models.components.binarization import (
    Binarize,
)

from eval_args import get_arg_parser
import constants
from vocab import Vocabulary, load_vocab
import dataset as d
import utils

def eval_bleu(
    train_loader: d.BatchedIterator,
    valid_loader: d.BatchedIterator,
    model: nn.Module,
    en_vocab: Vocabulary,
    fr_vocab: Vocabulary,
    device: str,
    multi_gpu: bool,
) -> None:
    model = model.to(device)
    if multi_gpu and device == 'cuda':
       print('Using multi gpu training')
       model = torch.nn.DataParallel(model, device_ids=[0, 1]).cuda()
    
    bleus = []
    count = 0
    with tqdm(train_loader, total=len(train_loader)) as pbar:
        for i, data in enumerate(pbar):
            if i == 0:
                continue
            src, src_lengths = data.src
            trg, trg_lengths = data.trg

            # predicted = model.generate_max(src, src_lengths, 100, device)
            predicted = model.slow_generate(src, src_lengths, 100, device)
            # predicted = (torch.Tensor(src.size(0), 100).uniform_() * (len(fr_vocab) - 1)).long()
            # predicted = predicted * 
            # predicted = model.generate_beam(src, src_lengths, 100, 5, device)
            pred_arr = utils.torchtext_convert_to_str(predicted.cpu().numpy(), fr_vocab)[0]
            out_arr = utils.torchtext_convert_to_str(trg.cpu().numpy(), fr_vocab)[0]
            pred_slim_arr = utils.get_raw_sentence(pred_arr)
            out_slim_arr = utils.get_raw_sentence(out_arr)
            curr_bleu = utils.compute_bleu(pred_slim_arr, out_slim_arr)
            # print("BLEU: {}".format(
            #     curr_bleu
            # ))
            bleus.append(curr_bleu)
            # output = ' '.join(pred_slim_arr)
            # actual_out = ' '.join(out_slim_arr)
            # src = ' '.join(utils.torchtext_convert_to_str(src.cpu().numpy(), en_vocab)[0])
            # print('src\n', src)
            # print('')
            # print('out\n',output)
            # print('')
            # print('trg\n', actual_out)

            # if (i >= 7):
            #     print(bleus)
            #     print(sum(bleus) / len(bleus))
            #     return

            count += 1
            pbar.set_postfix(
                curr_bleu=curr_bleu * 100,
                avg_bleu=(sum(bleus) / len(bleus) * 100)
            )
            pbar.refresh()

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
    model.load_state_dict(torch.load(args.load_path))
    model = model.eval()
    if args.binarize:
        print('binarizing model')
        binarized_model = Binarize(model)
        binarized_model.binarization()


    print('using model...')
    print(model)

    eval_bleu(
        train_loader=train_loader,
        valid_loader=None, # valid_loader,
        model=model,
        en_vocab=src.vocab,
        fr_vocab=trg.vocab,
        device=device,
        multi_gpu=args.multi_gpu,
    )

if __name__ == "__main__":
    main()