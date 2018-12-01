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
from train_args import get_arg_parser
import constants
from vocab import Vocabulary, load_vocab
import dataset as d

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
    else:
        raise Exception(
            "Unknown Model Type: {}".format(args.model_type)
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
                if i == 0:
                    continue
                src, trg, src_lengths, trg_lengths, prev_tokens, prev_lengths = data
                src = src.to(device).long()
                trg = trg.to(device).long()
                src_lengths = src_lengths.to(device).long()
                trg_lengths = trg_lengths.to(device)

                predicted = model.generate_max(src, src_lengths, 100, device)
                # predicted = model.generate_beam(src, src_lengths, 100, 5, device)
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


    print('loading datasets...')
    src = data.Field(include_lengths=True,
               init_token='<sos>', eos_token='<eos>', batch_first=True, fix_length=200)
    trg = data.Field(include_lengths=True,
               init_token='<sos>', eos_token='<eos>', batch_first=True)
    
    if not args.small:
        mt_train = datasets.TranslationDataset(
            path=constants.WMT14_EN_FR_SMALL_TRAIN,
            exts=('.en', '.fr'),
            fields=(src, trg)
        )
    else:
        mt_train, _, _ = datasets.Multi30k.splits(
            exts=('.en', '.de'),
            fields=(src, trg),
        )

    print('loading vocabulary...')
    src.build_vocab(mt_train, min_freq=2, max_size=80000)
    trg.build_vocab(mt_train, max_size=40000)
    print('loaded vocabulary')
    # mt_dev shares the fields, so it shares their vocab objects

    train_loader = data.BucketIterator(
        dataset=mt_train,
        batch_size=args.batch_size,
        sort_key=lambda x: len(x.src), # data.interleave_keys(len(x.src), len(x.trg)),
        sort_within_batch=True,
        device=device
    )

    model = build_model(parser, src.vocab, trg.vocab)

    print('using model...')
    print(model)

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    
    if not os.path.exists(os.path.join(args.save_dir, args.model_name)):
        os.makedirs(os.path.join(args.save_dir, args.model_name))

    eval_bleu(
        train_loader=train_loader,
        valid_loader=None, # valid_loader,
        model=model,
        en_vocab=en_vocab,
        fr_vocab=fr_vocab,
        device=device,
        multi_gpu=args.multi_gpu,
    )

if __name__ == "__main__":
    main()