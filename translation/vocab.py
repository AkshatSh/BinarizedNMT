from collections import Counter
import os
import argparse
from typing import Callable, List
import nltk
import spacy
import pickle
from tqdm import tqdm

import constants

class Vocabulary(object):
    '''
    A class to store the vocabulary for a dataset,
    this contains a mapping of words to unique indexes, with methods
    to look up the counts of a word, the index of a word, and a reverse look up
    from index to a word
    '''
    def __init__(self):
        self.wordidx = {}
        self.idxword = {}
        self.counter = Counter()
        self.index = 0
    
    def __len__(self):
        # the number of words in the vocabulary
        return len(self.wordidx)
    
    def word2idx(self, word:str) -> int:
        # return the idx for the word
        return self.wordidx[word] if word in self.wordidx else self.wordidx[constants.UNKNOWN_TOKEN]
    
    def idx2word(self, idx: str) -> str:
        # return the word for the idx
        return self.idxword[idx]
    
    def count(self, word: str) -> int:
        # returns the popularity of the word
        return self.counter[word] if word in counter else self.counter[constants.UNKNOWN_TOKEN]
    
    def add(self, word: str) -> None:
        # adds the word to the vocabulary
        self._add_no_count(word)
        self.counter.update([word])
    
    def _add_no_count(self, word: str) -> None:
        # does not update the counter for adding a word
        if word not in self.wordidx:
            self.wordidx[word] = self.index
            self.idxword[self.index] = word
            self.index += 1

    def unk(self, cutoff: int) -> int:
        '''
        Unks the vocabulary
        * Removes all words that occurr less than the cutoff,
        * returns the number of words removed from the vocabulary
        '''
        remove_list = []
        for word in self.counter:
            count = self.counter[word]
            if count < cutoff and word not in constants.SPECIAL_TOKENS:
                remove_list.append(word)
        
        for word in remove_list:
            del self.counter[word]
        
        self.idx = 0
        self.wordidx = {}
        self.idxword = {}
        for word in self.counter:
            self._add_no_count(word)

        return len(remove_list)

def build_vocab(
    data: List[object],
    item_fn: Callable[[object], str],
    token_fn: Callable[[str], List[str]] = lambda x : x.split(),
    unk_cutoff: int = None
):
    '''
    Given some list of objects or iteratable (data)
    applys the item_fn on each item in data to get a sentence
    tokenizes the sentence by token_fn,
    and optionally unks the vocabulary with the specified cutoff (None to void)
    '''
    vocab = Vocabulary()

    for item in tqdm(data):
        sentence = item_fn(item)
        tokens = token_fn(sentence)
        for token in tokens:
            vocab.add(token)
    if unk_cutoff is not None:
        vocab.unk(unk_cutoff)

    for token in constants.SPECIAL_TOKENS:
        vocab.add(token)
    return vocab

def load_vocab(
    file_name: str,
) -> Vocabulary:
    '''
    loads the saved state of a vocabulary object
    '''
    with open(file_name, 'rb') as f:
        vocab = pickle.load(f)
    return vocab

def save_vocab(
    vocab: Vocabulary,
    file_name: str,
) -> None:
    '''
    saves the state of a vocabulary object
    '''
    with open(file_name, 'wb') as f:
        pickle.dump(vocab, f)
