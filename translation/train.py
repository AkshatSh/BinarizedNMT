import torch
import vocab
import constants 
import conll2003dataloader
import pickle
import argparse
import os
import models.bilstm_crf as bilstm_crf
import torch
from train_args import get_arg_parser
from tensor_logger import Logger

from tqdm import trange, tqdm

def main(args):
    pass

if __name__ == "__main__":
    parser = get_arg_parser()
    args = parser.parse_args()
    main(args)