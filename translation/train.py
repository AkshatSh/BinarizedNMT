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

from models import SimpleLSTMModel
from train_args import get_arg_parser
import constants
from vocab import Vocabulary, load_vocab
import dataset as d

def build_model(parser, en_vocab, fr_vocab):
    # TODO make switch case
    SimpleLSTMModel.add_args(parser)
    args = parser.parse_args()
    return SimpleLSTMModel.build_model(
        en_vocab=en_vocab,
        fr_vocab=fr_vocab,
        encoder_embed_dim=args.encoder_embed_dim,
        encoder_hidden_dim=args.encoder_hidden_dim,
        encoder_dropout=args.encoder_dropout,
        decoder_embed_dim=args.decoder_embed_dim,
        decoder_hidden_dim=args.decoder_hidden_dim,
        decoder_dropout=args.decoder_dropout,
    )

def train(
    train_loader: d.BatchedIterator,
    valid_loader: d.BatchedIterator,
    model: nn.Module,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    log_dir: str,
    save_dir: str,
    en_vocab: Vocabulary,
    fr_vocab: Vocabulary,
    device: str,
):
    total_loss = 0.0
    count = 0
    model = model.to(device)
    #if device == 'cuda':
    #    print('Using multi gpu training')
    #    model = torch.nn.DataParallel(model, device_ids=[0, 1]).cuda()
    optim = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    for e in range(epochs):
        with tqdm(train_loader, total=len(train_loader)) as pbar:
            for src, trg, src_lengths, trg_lengths in pbar:
                src = src.to(device)
                trg = trg.to(device)
                src_lengths = src_lengths.to(device)
                trg_lengths = trg_lengths.to(device)
                # add EOS to trg
                # move EOS to front for prev
                # feed everything into model
                # compute loss
                # call backwards
                eos_tensor = torch.zeros(
                    (trg.shape[0], 1)
                ).fill_(
                    fr_vocab.word2idx(constants.END_TOKEN)
                ).long().to(device)

                trg_tensor = torch.cat([trg, eos_tensor], dim=1).to(device)
                prev_tokens = torch.cat([eos_tensor, trg], dim=1).to(device)
                predicted, _ = model.forward(src, src_lengths, prev_tokens)
                loss = model.loss(predicted.view(-1, predicted.size(-1)), trg_tensor.view(-1))
                # loss = F.cross_entropy(
                #     predicted.view(-1, predicted.size(-1)),
                #     trg_tensor.view(-1),
                #     ignore_index=fr_vocab.word2idx(constants.PAD_TOKEN),
                # )
                loss.backward()
                optim.step()
                total_loss += loss.item()
                count += 1
                pbar.set_postfix(
                    loss_avg=total_loss/(count),
                    epoch="{}/{}".format(e + 1, epochs),
                    curr_loss=loss.item(),
                )
                pbar.refresh()
 
        train_loader.reset()
        valid_loader.reset()

def main():
    parser = get_arg_parser()
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() and args.cuda else "cpu"
    print('using device {}'.format(device))

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

    model = build_model(parser, en_vocab, fr_vocab)

    print('using model...')
    print(model)

    train(
        train_loader=train_loader,
        valid_loader=valid_loader,
        model=model,
        epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        log_dir=args.log_dir,
        save_dir=args.save_dir,
        en_vocab=en_vocab,
        fr_vocab=fr_vocab,
        device=device,
    )

if __name__ == "__main__":
    main()