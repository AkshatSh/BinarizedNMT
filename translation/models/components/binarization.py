'''
This module implements the binarization of a 1 dimensional convolutional model.

Based on the XNOR net paper: https://arxiv.org/pdf/1603.05279.pdf

Implementation: https://github.com/jiecaoyu/XNOR-Net-PyTorch

The paper and implementation target 2 dimensional convolutions for image classification tasks (MNIST, CIFAR, Imagenet)
where as here we target 1 dimensional convolutions for NLP tasks in particular Neural Machine Translation.

'''
import sys
import math
import torch.nn as nn
import torch
import torch.nn.functional as F

class Binarize(Object):
    '''
    This object wraps the model passed in, to allow binary network training
    as described in the xnor paper: Algorithm 1

    It cycles through all the modules that the model is using,
    and targets every nn.Conv1D being used.

    It supports the following methods
        binarization(self):
            converts all the weights to binarized versions
        restore(self):
            resets the convolution weights to what they were
            before binarization
        updateGradients(self):
            Updates the gradients using the estimated binarized weights

    ex:
        binarized_model = Binarize(model)
        for ....
            binarized_model.binarization()

            # compute loss
            model.loss(...)

            binarized_model.restore()
            binarized_model.updateGradients()

            # update the actual model
            optimizer.step()
    '''
    def __init__(
        self,
        model: nn.Module,
    ):
        super(Binarize, self).__init__()
        self.model = model