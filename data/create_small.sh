#!/bin/bash

# since our dataset is quite large (40 million examples), to do rapid development,
# we can use a small dataset (4 million examples) and the same valid and test set

# exit immediately if any command fails
set -e

# arguments
size="4000000"
if [ "$#" -eq 1 ]; then
  size="$1"
fi

head -n "$size" wmt14_en_fr/train.en > wmt14_en_fr/small_train.en
head -n "$size" wmt14_en_fr/train.fr > wmt14_en_fr/small_train.fr