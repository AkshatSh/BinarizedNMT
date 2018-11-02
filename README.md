# BinarizedNMT

## Proposal

Exploring ways to reduce Neural Machine Translation Memory and Computation through binarized networks

With the development of binarized networks showing promise in vision, such as xnor-net [1], and promise in language modeling [3]. We want to use our deep learning project to investigate how we can use binarized networks to reduce memory use and computation for machine translation, a possible application being translation locally on edge devices.

### Approach

Following the development of xnor-net we aim to use the binarized weights and convolutions in place of the normal convolutions in convolution based machine translation networks such as Conv2Seq [4] and Quasi Recurrent Networks [5] and compare the memory use and computation to the non binarized equivalent and state of the art translation models.

### Dataset

To make the scope a bit more manageable we will specifically look at English to French translation [6].

### Training

We will implement the models in PyTorch and train these models on a mixture of colab machines, attu gpu machines, and personal computers.

## Implementation

### Components to be Implemented

These are the list of components to be implemented that will be useful in all the differnet models we will use.

#### Binarized Convolution

This is the convolution introduced in the xnor-net paper for binary convolution, this is what will do the majority of the space saving and computation saving. The paper [1] has the definition of the component mathematically and [2] is a pytorch implementation of the xnor-net.

- [] Implement Transformer
- [] Add tests

#### Self Attention

Self attention is useful for most of these networks so the network can learn that some words are more importnat than the other. The majority of the network implementations have them defined in the network, abstracting it away will help us keep things consistent.

#### Multihead Attention

This was defined in the attention is all you need paper [7]. If we implement the attention transformer this will be useful. An annotated guide for the transformer is [3].

### Models

#### Seq2Seq

Following the implementation of the Seq2Seq model, and Seq2Seq with attention, we want to experiment how a simple LSTM based sequence model handles our Machine Translation task as a baseline benchmark.

- [ ] Adjust Pytorch tutorial for our case
- [ ] Tune Model for optimal performance
- [ ] Compare runtime of inference on CPU and GPU
- [ ] Get metric Benchmarks

#### QRNN (Quasi Recurrent Neural Networks)

A network developed by Salesforce Research which replaces LSTM cells in recurrent networks with convolutional networks, making the network much more parallel. We wil investigate a network that can replace these convlutions with binary convolutions. Implementation of the paper by the researchers is in [4] and the paper itself in [5] (said to be 17x faster than cudNN LSTM).

- [ ] Adjust QRNN to work for our case
- [ ] Implement Binarized QRNN
- [ ] Tune Binraized QRNN
- [ ] Compare runtime of inference on CPU and GPU
- [ ] Get metric Benchmarks


#### Convolutional Sequence Learning

A network developed by Facebook Research (FAIR) which does not rely on RNNs but rather an encoder decoder model with convolutions instead. This has the most promise since it relies the most on convolutions. The paper [4] and github [5] have the materials necessary.

- [ ] Adjust ConvSeq to work for our case
- [ ] Implement Binarized ConvSeq
- [ ] Tune Binraized ConvSeq
- [ ] Compare runtime of inference on CPU and GPU
- [ ] Get metric Benchmarks

#### Attention Transformer

If we have time, this is state of the art which is highly paralleized but trains on Googles TPUs, which require a lot memory, can we implement a biniarized version of this.

### Metircs to Analyze

#### CPU Time and Computation

#### GPU Time and Computation

#### BLEU Score

#### Size of Model

## References

### Papers

1. Xnor - Net: [Paper](https://arxiv.org/abs/1603.05279)
2. Multi bit quantization networks: [Paper](https://arxiv.org/pdf/1802.00150.pdf)
3. Binarized LSTM Language Model: [Paper](http://aclweb.org/anthology/N18-1192)
4. Fair Seq Convolutinal Sequence Learning: [Paper](https://arxiv.org/pdf/1705.03122.pdf)
5. Quasi Recurrent Networks: [Paper](https://arxiv.org/abs/1611.01576 )
6. WMT 14 Translation Task [Paper](http://www.statmt.org/wmt14/translation-task.html)
7. Attention is all you need [Paper](https://arxiv.org/abs/1706.03762)

### Githubs and Links

1. [Pytorch MT Seq2Seq Tutorial](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)
2. [Xnor NET Pytorch](https://github.com/jiecaoyu/XNOR-Net-PyTorch)
3. [Annotated Transformer (Harvard NLP)](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
4. [Salesforce QRNN Pytorch](https://github.com/salesforce/pytorch-qrnn)
5. [Fair Seq](https://github.com/pytorch/fairseq)
