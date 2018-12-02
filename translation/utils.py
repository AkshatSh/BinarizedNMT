import os
import mmap
from typing import List
import torch
import numpy as np
import torchtext
from torchtext.vocab import Vocab
from typing import Tuple
import pickle
import nltk

from vocab import Vocabulary
import constants

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
    return nltk.translate.bleu_score.sentence_bleu([expected], predicted)