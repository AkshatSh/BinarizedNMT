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

#### Binarized Convolution


#### Self Attention

#### Multihead Attention

### Models

#### Seq2Seq

Following the implementation of the Seq2Seq model, and Seq2Seq with attention, we want to experiment how a simple LSTM based sequence model handles our Machine Translation task as a baseline benchmark.

- [ ] Adjust Pytorch tutorial for our case
- [ ] Tune Model for optimal performance
- [ ] Compare runtime of inference on CPU and GPU 
- [ ] Get metric Benchmarks

#### QRNN (Quasi Recurrent Neural Networks)


#### Convolutional Sequence Learning


#### Attention Transformer

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