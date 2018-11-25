import os
import mmap
from typing import List
import torch

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
    tensor: torch.Tensor,
    vocab: Vocabulary,
) -> List[List[str]]:
    output = []
    for batch in range(tensor):
        curr = []
        for idx in range(tensor[batch]):
            curr.append(vocab.idx2word(tensor[batch, idx]))
        output.append(curr)
    return output