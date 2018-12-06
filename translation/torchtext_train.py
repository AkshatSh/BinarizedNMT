import torch
import pickle
import argparse
import os
from tqdm import trange, tqdm
import torch
import torchtext
from torchtext import data
from torchtext import datasets
from torchtext.vocab import Vocab
from torch import nn
import torch.nn.functional as F
import math

from models import (
    AttentionRNN,
    # ConvSeq2Seq,
    ConvS2S,
    SimpleLSTMModel,
)

from train_args import get_arg_parser
import constants
from vocab import Vocabulary, load_vocab
import dataset as d
import utils
from tensor_logger import Logger

def eval_model(
    src_vocab: Vocab,
    trg_vocab: Vocab,
    model: nn.Module,
    valid_loader: d.BatchedIterator,
) -> tuple:
    model.eval()
    total_loss = 0.0
    total_ppl = 0.0
    with torch.no_grad():
        for i, data in enumerate(tqdm(valid_loader)):
            src, src_lengths = data.src
            trg, trg_lengths = data.trg
            # feed everything into model
            # compute loss
            # compute ppl
            predicted, _ = model.forward(src, src_lengths, trg)
            loss = F.cross_entropy(
                predicted[:, :-1].contiguous().view(-1, len(trg_vocab)),
                trg[:, 1:].contiguous().view(-1),
                ignore_index=trg_vocab.stoi['<pad>'],
            )

            total_loss += loss.item()
            total_ppl += math.exp(loss.item())
    model.train()
    return total_loss / len(valid_loader), total_ppl / len(valid_loader)

def train(
    train_loader: d.BatchedIterator,
    valid_loader: d.BatchedIterator,
    model: nn.Module,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    log_dir: str,
    save_dir: str,
    en_vocab: Vocab,
    fr_vocab: Vocab,
    device: str,
    multi_gpu: bool,
    save_step: int,
    model_name: str,
    optimizer: str,
    batch_size: int,
    log_step: int,
    should_save: bool,
) -> None:
    logger = Logger(log_dir)
    model = model.to(device)
    if multi_gpu and device == 'cuda':
       print('Using multi gpu training')
       model = torch.nn.DataParallel(model, device_ids=[0, 1]).cuda()
    
    if optimizer == "sgd":
        print("using stochastic gradient descent optimizer")
        optim = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer == "adam":
        print("using adam optimizer")
        optim = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        raise Exception("Illegal Optimizer {}".format(optimizer))

    nan_count = 0
    for e in range(epochs):
        total_loss = 0.0
        count = 0
        with tqdm(train_loader, total=len(train_loader)) as pbar:
            for i, data in enumerate(pbar):
                src, src_lengths = data.src
                trg, trg_lengths = data.trg
                # feed everything into model
                # compute loss
                # call backwards
                optim.zero_grad()
                predicted, _ = model.forward(src, src_lengths, trg)
                if not multi_gpu:
                    loss = F.cross_entropy(
                        predicted[:, :-1].contiguous().view(-1, len(fr_vocab)),
                        trg[:, 1:].contiguous().view(-1),
                        ignore_index=fr_vocab.stoi['<pad>'],
                    )
                else:
                    # if using data parallel, loss has to be computed here
                    # there is no longer a model loss function that we have
                    # access to.
                    # TODO: data parallel kills the computer, why?
                    loss = F.cross_entropy(
                        predicted.view(-1, predicted.size(-1)),
                        trg_tensor.view(-1),
                        ignore_index=fr_vocab.word2idx(constants.PAD_TOKEN),
                    )

                # if loss.item() > 1e2:
                #     print('something happened at: {} with loss: {}'.format(i, loss.item()))
                #     torch.save(
                #         model.state_dict(), 
                #         os.path.join('exploding_problem.pt')
                #     )
                #     return
                if should_save and math.isnan(loss.item()):
                    '''
                    Ignore nan loss for backward, and continue forward
                    '''
                    nan_count += 1
                    print('found nan at {}'.format(i))
                    torch.save(
                        model.state_dict(), 
                        os.path.join(save_dir, model_name, 'unk_problem.pt')
                    )
                    return
                loss.backward()

                # TODO: try gradient clipping? for exploding gradient
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                # for p in model.parameters():
                #     p.data.add_(-learning_rate, p.grad.data)

                optim.step()
                total_loss += loss.item()
                count += 1
                pbar.set_postfix(
                    loss_avg=total_loss/(count),
                    epoch="{}/{}".format(e + 1, epochs),
                    curr_loss=loss.item(),
                    nan_count=nan_count,
                )
                pbar.refresh()

                if should_save and (i + 1) % log_step == 0:
                    # log every log step (excluding 0 for noise)
                    logger.scalar_summary(
                        "train loss_avg", 
                        total_loss/count,
                        (e * len(train_loader) + i) * batch_size,
                    )

                    logger.scalar_summary(
                        "loss", 
                        loss.item(),
                        (e * len(train_loader) + i) * batch_size,
                    )

                if (i + 1) % save_step == 0:
                    print('Saving model at iteration {} for epoch {}'.format(i, e))
                    model_file_name = "model_epoch_{}_itr_{}".format(e, i)
                    torch.save(
                        model.state_dict(), 
                        os.path.join(save_dir, model_name, model_file_name)
                    )
            print("Summary: Total Loss {} | Count {} | Average {}".format(total_loss, count, total_loss / count))
            if should_save:
                model_file_name = "model_epoch_{}_final".format(e)
                print('saving to {}'.format(os.path.join(save_dir, model_name, model_file_name)))
                torch.save(
                    model.state_dict(), 
                    os.path.join(save_dir, model_name, model_file_name)
                )
            if valid_loader is not None:
                valid_loss, valid_ppl = eval_model(en_vocab, fr_vocab, model, valid_loader)
                print("Valid loss avg : {} | Valid Perplexity: {}".format(valid_loss, valid_ppl))

def main() -> None:
    parser = get_arg_parser()
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() and args.cuda else "cpu"
    print('using device {}'.format(device))


    print('loading datasets...')
    src = data.Field(
        include_lengths=True,
        init_token='<sos>',
        eos_token='<eos>',
        batch_first=True,
        fix_length=args.torchtext_src_fix_length,
    )

    trg = data.Field(
        include_lengths=True,
        init_token='<sos>',
        eos_token='<eos>',
        batch_first=True,
    )
    
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
            mt_train, mt_valid, _ = datasets.Multi30k.splits(
                exts=('.en', '.de'),
                fields=(src, trg),
            )
        elif args.dataset == 'IWSLT':
            mt_train, mt_valid, _ = datasets.IWSLT.splits(
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

    train_loader = data.BucketIterator(
        dataset=mt_train,
        batch_size=args.batch_size,
        sort_key=lambda x: len(x.src), # data.interleave_keys(len(x.src), len(x.trg)),
        sort_within_batch=True,
        device=device,
    )

    valid_loader = None if mt_valid is None else data.BucketIterator(
        dataset=mt_valid,
        batch_size=args.batch_size,
        sort_key=lambda x: len(x.src), # data.interleave_keys(len(x.src), len(x.trg)),
        sort_within_batch=True,
        device=device,
    )

    model = utils.build_model(parser, src.vocab, trg.vocab)
    # model.load_state_dict(torch.load('saved_models/conv_test_large/model_epoch_9_final'))

    print('using model...')
    print(model)

    log_dir = os.path.join(
        args.log_dir,
        args.model_name,
    )
    if args.should_save and  not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    if args.should_save and not os.path.exists(os.path.join(args.save_dir, args.model_name)):
        os.makedirs(os.path.join(args.save_dir, args.model_name))
    train(
        train_loader=train_loader,
        valid_loader=valid_loader,
        model=model,
        epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        log_dir=log_dir,
        save_dir=args.save_dir,
        en_vocab=src.vocab,
        fr_vocab=trg.vocab,
        device=device,
        multi_gpu=args.multi_gpu,
        save_step=args.save_step,
        model_name=args.model_name,
        optimizer=args.optimizer,
        batch_size=args.batch_size,
        log_step=args.log_step,
        should_save=args.should_save,
    )

if __name__ == "__main__":
    main()