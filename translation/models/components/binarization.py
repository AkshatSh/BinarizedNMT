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

from .binarized_convolution import (
    BinConv1d,
)

class Binarize(object):
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
            loss = model.loss(...)
            loss.backward()

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

        # get the number of conv1d modules
        conv1d_count = 0
        for m in model.modules():
            if isinstance(m, nn.Conv1d):
                conv1d_count += 1
        
        start_range = 0
        end_range = conv1d_count
        self.bin_range = [i for i in range(start_range, end_range)]
        self.num_of_params = len(self.bin_range)
        self.saved_params = []
        self.target_params = []
        self.target_modules = []
        index = 0
        for m in model.modules():
            if isinstance(m, BinConv1d):
                # save the weight
                m = m.conv
                saved_weight = m.weight.data.clone()
                self.saved_params.append(saved_weight)
                self.target_modules.append(m.weight)

                # weight has the shape:
                # (out_channel, in_channel, kernel_size)
        print("Targeting {} convolutions.".format(len(self.target_modules)))
    
    def meanCenterConvParams(self):
        for index in range(len(self.target_modules)):
            s = self.target_modules[index].data.size()
            negMean = self.target_modules[index].data.mean(1, keepdim=True).\
                    mul(-1).expand_as(self.target_modules[index].data)
            self.target_modules[index].data = self.target_modules[index].data.add(negMean)
    
    def clampConvParams(self):
        for index in range(len(self.target_modules)):
            self.target_modules[index].data = \
                    self.target_modules[index].data.clamp(-1.0, 1.0)
    
    def save_params(self):
        for index in range(len(self.target_modules)):
            self.saved_params[index].copy_(self.target_modules[index].data)
    
    def binarize_conv_params(self):
        for index in range(len(self.target_modules)):
            # n = kernel_size * in_channels
            curr_module = self.target_modules[index].data
            n = curr_module[0].nelement()
            s = curr_module.size()

            # abs mean normalizes every filter and divides by n to get the 
            # normalized mean
            abs_mean = curr_module.norm(1, 2, keepdim=True).sum(1, keepdim=True).div(n).expand(s)

            # binarized weight is
            # sign(W) * abs_mean
            self.target_modules[index].data = curr_module.sign().mul(abs_mean)

    def binarization(self):
        self.meanCenterConvParams()
        self.clampConvParams()
        self.save_params()
        self.binarize_conv_params()
    
    def restore(self):
        for index in range(len(self.target_modules)):
            self.target_modules[index].data.copy_(self.saved_params[index])
    
    def updateGradients(self):
        for index in range(len(self.target_modules)):
            curr_module = self.target_modules[index].data
            n = curr_module[0].nelement()
            s = curr_module.size()

            m = curr_module.norm(1, 2, keepdim=True).sum(1, keepdim=True).div(n).expand(s)
            m[curr_module.lt(-1.0)] = 0 
            m[curr_module.gt(1.0)] = 0

            m = m.mul(self.target_modules[index].grad.data)
            m_add = curr_module.sign().mul(self.target_modules[index].grad.data)
            m_add = m_add.sum(2, keepdim=True)\
                    .sum(1, keepdim=True).div(n).expand(s)
            m_add = m_add.mul(curr_module.sign())

            self.target_modules[index].grad.data = m.add(m_add).mul(1.0-1.0/s[1]).mul(n)