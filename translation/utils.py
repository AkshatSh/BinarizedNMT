import os
import mmap
from typing import List
import torch
import numpy as np

from vocab import Vocabulary

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