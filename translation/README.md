# Setup

```bash

# small train
$ python translation/shard_dataset.py --src_file data/wmt14_en_fr/small_train.en --target_file data/wmt14_en_fr/small_train.fr --output_dir data/wmt14_en_fr/small_train_shard

# large train
$ python translation/shard_dataset.py --src_file data/wmt14_en_fr/train.en --target_file data/wmt14_en_fr/train.fr --output_dir data/wmt14_en_fr/train_shard

# small vocab
$ python translation/dataset.py --small

# normal vocab
$ python translation/dataset.py
```