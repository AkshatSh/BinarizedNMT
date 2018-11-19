import os
import sys

WMT14_EN_FR_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'wmt14_en_fr/')
WMT14_EN_FR_TRAIN = os.path.join(WMT14_EN_FR_DIR, 'train')
WMT14_EN_FR_VALID = os.path.join(WMT14_EN_FR_DIR, 'valid')
WMT14_EN_FR_TEST = os.path.join(WMT14_EN_FR_DIR, 'test')
WMT14_EN_FR_TRAIN_SHARD = os.path.join(WMT14_EN_FR_DIR, 'train_shard')

UNKNOWN_TOKEN = "<UNK>"
START_TOKEN = "<START>"
PAD_TOKEN = "<PAD>"
END_TOKEN = "<END>"
SPECIAL_TOKENS = [PAD_TOKEN, UNKNOWN_TOKEN, START_TOKEN, END_TOKEN]

TRAIN_EN_VOCAB_FILE = os.path.join(WMT14_EN_FR_DIR, 'train_en_vocab.pkl')
TRAIN_FR_VOCAB_FILE = os.path.join(WMT14_EN_FR_DIR, 'train_fr_vocab.pkl')