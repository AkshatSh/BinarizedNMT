#!/bin/bash
# To set up environment

# exit immediately if any command fails
set -e

# create virtual env
DIR_ENV="./env"
if [ ! -d "$DIR_ENV" ]; then
  echo "Creating virtual environment ..."
  python3 -m venv env
  source env/bin/activate
  pip install -r requirements.txt
fi

# download dataset
DATA_DIR="./data-bin"
if [ ! -d "$DATA_DIR" ]; then
  echo "Downloading WMT 14 english to french dataset ..."
  wget https://s3.amazonaws.com/fairseq-py/data/wmt14.v2.en-fr.newstest2014.tar.bz2
  mkdir data-bin
  tar -xvjf wmt14.v2.en-fr.newstest2014.tar.bz2 -C data-bin
  rm wmt14.v2.en-fr.newstest2014.tar.bz2
  cd data
  bash prepare.sh
  cd ..
  python translation/shard_dataset.py
  python translation/dataset.py
fi
