import sys
sys.path.append("..")

import torch
from torch import nn
import torch.nn.functional as F
import random
import argparse

from models.EncoderDecoder import (
    EncoderModel,
    DecoderModel,
    EncoderDecoderModel,
    DecoderOutputType,
)

from models.components.attention import (
    AttentionModule
)

from vocab import Vocabulary

from constants import (
    UNKNOWN_TOKEN,
    PAD_TOKEN,
)

'''
Implementation of the convolutional sequence to sequence architecture described here:
https://arxiv.org/pdf/1705.03122.pdf 


Adopted from: Fair Seq https://github.com/pytorch/fairseq/blob/master/fairseq/models/fconv.py
'''