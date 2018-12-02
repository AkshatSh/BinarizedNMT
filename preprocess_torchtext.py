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
from typing import Tuple

from translation import constants

# dataset, src_field, trg_field
LoadedDatasetType = Tuple[datasets.TranslationDataset, data.Field, data.Field]

# src_vocab, trg_vocab
TranslationVocabularyType = Tuple[Vocab, Vocab]

def load_wmt_small_dataset() -> LoadedDatasetType:
    src = data.Field(include_lengths=True,
               init_token='<sos>', eos_token='<eos>', batch_first=True, fix_length=200)
    trg = data.Field(include_lengths=True,
               init_token='<sos>', eos_token='<eos>', batch_first=True)

    mt_train = datasets.TranslationDataset(
        path=constants.WMT14_EN_FR_SMALL_TRAIN,
        exts=('.en', '.fr'),
        fields=(src, trg)
    )

    return mt_train, src, trg

def load_vocabulary(
    dataset_info: LoadedDatasetType,
) -> TranslationVocabularyType:
    mt_train, src, trg = dataset_info
    src.build_vocab(mt_train, min_freq=2, max_size=80000)
    trg.build_vocab(mt_train, max_size=10000)
    return src.vocab, trg.vocab

def main() -> None:
    print('[INFO]: loading dataset ...')
    dataset_info = load_wmt_small_dataset()
    print('[INFO]: loaded dataset')

    print('[INFO]: building vocabulary ...')
    src_vocab, trg_vocab = load_vocabulary(dataset_info)
    print('[INFO]: built vocabulary')

    print('[INFO]: saving vocabulary ...')
    with open(constants.TORCH_TEXT_SMALL_EN_VOCAB_FILE, 'wb') as f:
        pickle.dump(src_vocab, f)
    
    with open(constants.TORCH_TEXT_SMALL_FR_VOCAB_FILE, 'wb') as f:
        pickle.dump(trg_vocab, f)
    print('[INFO]: saved vocabulary')

    print('[INFO]: done.')

if __name__ == "__main__":
    main()

