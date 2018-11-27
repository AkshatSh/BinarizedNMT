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

from models import SimpleLSTMModel
from eval_args import get_arg_parser
import constants
from vocab import Vocabulary, load_vocab
import dataset as d
import utils

def build_model(
    parser: argparse.ArgumentParser,
    en_vocab: Vocabulary,
    fr_vocab: Vocabulary,
) -> nn.Module:
    # TODO make switch case
    SimpleLSTMModel.add_args(parser)
    args = parser.parse_args()
    return SimpleLSTMModel.build_model(
        en_vocab=en_vocab,
        fr_vocab=fr_vocab,
        encoder_embed_dim=args.encoder_embed_dim,
        encoder_hidden_dim=args.encoder_hidden_dim,
        encoder_dropout=args.encoder_dropout,
        encoder_num_layers=args.encoder_num_layers,
        decoder_embed_dim=args.decoder_embed_dim,
        decoder_hidden_dim=args.decoder_hidden_dim,
        decoder_dropout=args.decoder_dropout,
        decoder_num_layers=args.decoder_num_layers,
    )

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
    
    for e in range(1):
        count = 0
        with tqdm(train_loader, total=len(train_loader)) as pbar:
            for i, data in enumerate(pbar):
                src, trg, src_lengths, trg_lengths, prev_tokens, prev_lengths = data
                src = src.to(device).long()
                trg = trg.to(device).long()
                src_lengths = src_lengths.to(device).long()
                trg_lengths = trg_lengths.to(device)

                # predicted = model.generate_max(src, src_lengths, 100, device)
                predicted = model.generate_beam(src, src_lengths, 100, 5, device)
                output = ' '.join(utils.convert_to_str(predicted.cpu().numpy(), fr_vocab)[0])
                actual_out = ' '.join(utils.convert_to_str(trg.cpu().numpy(), fr_vocab)[0])
                src = ' '.join(utils.convert_to_str(src.cpu().numpy(), en_vocab)[0])
                print('src\n', src)
                print('')
                print('out\n',output)
                print('')
                print('trg\n', actual_out)

                if (i >= 2):
                    return

                count += 1
                # pbar.set_postfix(
                #     loss_avg=total_loss/(count),
                #     epoch="{}/{}".format(e + 1, epochs),
                #     curr_loss=loss.item(),
                #     nan_count=nan_count,
                # )
                pbar.refresh()
 
        train_loader.reset()
        valid_loader.reset()

def main() -> None:
    parser = get_arg_parser()
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() and args.cuda else "cpu"
    print('using device {}'.format(device))

    print('loading vocabulary...')
    if args.small:
        print('using small training set')
        en_vocab = load_vocab(constants.SMALL_TRAIN_EN_VOCAB_FILE)
        fr_vocab = load_vocab(constants.SMALL_TRAIN_FR_VOCAB_FILE)
    else:
        en_vocab = load_vocab(constants.TRAIN_EN_VOCAB_FILE)
        fr_vocab = load_vocab(constants.TRAIN_FR_VOCAB_FILE)
    print('loaded vocabulary')

    print('loading datasets...')
    if args.small:
        train_dataset = d.ShardedCSVDataset(constants.WMT14_EN_FR_SMALL_TRAIN_SHARD)
    else:
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

    model = build_model(parser, en_vocab, fr_vocab)
    model.load_state_dict(torch.load(args.load_path))

    model = model.eval()
    eval_bleu(
        train_loader=train_loader,
        valid_loader=valid_loader,
        model=model,
        en_vocab=en_vocab,
        fr_vocab=fr_vocab,
        device=device,
        multi_gpu=args.multi_gpu,
    )

if __name__ == "__main__":
    main()