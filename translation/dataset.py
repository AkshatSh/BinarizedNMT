import os
import csv
import torch
import torchtext
import torchvision.transforms as transforms
import torch.utils.data as data
import random
from typing import Tuple, Callable

from vocab import Vocabulary, save_vocab, build_vocab, load_vocab
from utils import get_num_lines
from constants import WMT14_EN_FR_TRAIN_SHARD, TRAIN_EN_VOCAB_FILE, TRAIN_FR_VOCAB_FILE, UNKNOWN_TOKEN

# stores the entry src, trg for
# translation
Entry_Type = Tuple[str, str]

class ShardedDataset(object):
    def __init__(self):
        raise NotImplementedError()
    
    def __len__(self) -> int:
        raise NotImplementedError()
    
    def __iter__(self):
        return self
    
    def __next__(self) -> Entry_Type:
        raise NotImplementedError()

class ShardedCSVDataset(ShardedDataset):
    '''
    This dataset assumes that a larger dataset has been broken apart into
    a series of shards. All of which are defined as csv files, in the format
    src_line, translated_line.

    This dataset will retrieve each of the lines

    Arguments:
        sharded_dir: the directory with all the shards
    '''
    def __init__(self, sharded_dir: str, shuffle: bool = False):
        self.sharded_dir = sharded_dir
        self.shuffle = shuffle
        self.sharded_files = os.listdir(self.sharded_dir)

        self.sharded_files = [
            os.path.join(self.sharded_dir, file_name) for file_name in self.sharded_files
        ]

        if shuffle:
            random.shuffle(self.sharded_files)

        self.file_counts = {
            file_name : get_num_lines(file_name)
            for file_name in self.sharded_files
        }

        self.curr_file = None
        self.curr_file_name = None
        self.curr_idx = 0
        self.curr_data = []

        self.shard_idx = -1
    
    def __len__(self):
        return sum(self.file_counts.values())
    
    def reset(self):
        self.curr_file = None
        self.curr_file_name = None
        self.curr_idx = 0
        self.curr_data = []
        self.shard_idx = -1
    
    def __next__(self) -> Entry_Type:
        if self.curr_file is None or self.curr_idx >= self.file_counts[self.curr_file_name]:
            # read the new file into memory
            self.curr_file.close() if self.curr_file is not None else None

            self.shard_idx += 1

            if (self.shard_idx >= len(self.sharded_files)):
                # reached the end of the dataset
                raise StopIteration

            file_name = self.sharded_files[self.shard_idx]
            self.curr_file = open(file_name)
            self.curr_file_name = file_name
            self.curr_idx = 0
            
            # read the file into the dataset
            reader = csv.reader(self.curr_file, delimiter=',')
            self.curr_data = [(row[0], row[1]) for row in reader]

            if self.shuffle:
                random.shuffle(self.curr_data)
        
        # return the current line (src,  trg)
        val = self.curr_data[self.curr_idx]
        self.curr_idx += 1
        return val

class DualFileDataset(object):
    '''
    A dataloader iterator that  goes over two files, where each line
    in the src_file corresponds to the label in the trg_file

    src_file and trg_file must have the same lengths

    Will randomize the dataset if shuffle is set to true
    '''
    def __init__(self, src_file: str, trg_file: str, shuffle: bool = False):
        self.src_file_name = src_file
        self.trg_file_name = trg_file
        self.shuffle = shuffle
        self.reset()
    
    def __len__(self) -> int:
        '''
        Will give back the length of the data objects
        '''
        return len(self.data)
    
    def reset(self) -> None:
        '''
        Resets the state of the iterator by re-reading from disk
        '''
        src_file = open(self.src_file_name)
        trg_file = open(self.trg_file_name)

        src_file_length = get_num_lines(self.src_file_name)
        trg_file_length = get_num_lines(self.trg_file_name)

        assert(src_file_length == trg_file_length)

        data = []
        for src, trg in zip(src_file, trg_file):
            src = src.strip()
            trg = trg.strip()

            if src and trg:
                data.append((src, trg))

        self.idx = 0
        self.data = data

        if self.shuffle:
            random.shuffle(self.data)

        src_file.close()
        trg_file.close()
    
    def __next__(self) -> Entry_Type:
        if self.idx >= len(self.data):
            raise StopIteration
        
        val = self.data[self.idx]

        self.idx += 1
        return val


class BatchedIterator(object):
    '''
    This is a batched iterator for the sharded dataset object.
    Given a batch size and english and french vocab, it returns a tensor
    of size (batch x max_sequence_length) for english in src and french in trg
    '''
    def __init__(
        self, 
        batch_size: int,
        data: ShardedDataset,
        en_vocab: Vocabulary,
        fr_vocab: Vocabulary,
        max_sequence_length: int,
        tokenize_fn: Callable[[str], str] = lambda x: x.split(),
    ):
        self.batch_size = batch_size
        self.data = data
        self.en_vocab = en_vocab
        self.fr_vocab = fr_vocab
        self.max_sequence_length = max_sequence_length
        self.tokenize_fn = tokenize_fn
    
    def __len__(self):
        '''
        Number of batches in the dataset
        '''
        return len(self.data) / self.batch_size
    
    def __next__(self) -> torch.Tensor:
        count = 0
        curr = next(self.data, None)

        if curr is None:
            raise StopIteration

        src = []
        trg = []

        while curr is not None and count < self.batch_size:
            src.append(self.tokenize_fn(curr[0]))
            trg.append(self.tokenize_fn(curr[1]))
            curr = next(self.data, None)
            count += 1
        
        indexes = [i for i in range(len(src))]
        indexes.sort(key=lambda i: len(src[i]), reverse=True)
        indexes = torch.Tensor(indexes).long()
        src_lengths = torch.LongTensor([len(item) for item in src])
        trg_lengths = torch.LongTensor([len(item) for item in trg])

        # res has a series of tuples that are the src and the output
        src_tensor = torch.Tensor(
            count,
            max(src_lengths),
        ).long().fill_(self.en_vocab.word2idx(UNKNOWN_TOKEN))

        trg_tensor = torch.Tensor(
            count,
            max(trg_lengths),
        ).long().fill_(self.fr_vocab.word2idx(UNKNOWN_TOKEN))

        for i in range(count):
            src_tensor[i][:len(src[i])] = torch.LongTensor([
                self.en_vocab.word2idx(word) for word in src[i]
            ])[:len(src[i])]

            trg_tensor[i][:len(trg[i])] = torch.LongTensor([
                self.fr_vocab.word2idx(word) for word in trg[i]
            ])[:len(trg[i])]
        
        return (
            torch.index_select(src_tensor, 0, indexes),
            torch.index_select(trg_tensor, 0, indexes), 
            torch.index_select(src_lengths, 0, indexes),
            torch.index_select(trg_lengths, 0, indexes),
        )

def save_shard_vocab() -> None:
    '''
    Reads the shard dataset and creates vocabulary objects
    '''
    d = ShardedCSVDataset(WMT14_EN_FR_TRAIN_SHARD)

    def en_fn(item):
        return item[0]

    def fr_fn(item):
        return item[1]

    en_vocab = build_vocab(
        d,
        en_fn,
        unk_cutoff=2,
    )

    d.reset()

    fr_vocab = build_vocab(
        d,
        fr_fn,
        unk_cutoff=2,
    )

    save_vocab(en_vocab, TRAIN_EN_VOCAB_FILE)
    save_vocab(fr_vocab, TRAIN_FR_VOCAB_FILE)

def test() -> None:
    en_vocab = load_vocab(TRAIN_EN_VOCAB_FILE)
    fr_vocab = load_vocab(TRAIN_FR_VOCAB_FILE)

    print(
        "English Vocab Size: {} French Vocab Size: {}".format(len(en_vocab), len(fr_vocab))
    )

if __name__ == "__main__":
    save_shard_vocab()
    test()