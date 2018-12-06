'''
This is from the Pytorch implementation of XNOR-NET (will replace with our own)

Implementation is linked here: https://github.com/jiecaoyu/XNOR-Net-PyTorch/blob/master/CIFAR_10/models/nin.py
'''

import torch.nn as nn
import torch
import torch.nn.functional as F

class BinActive(torch.autograd.Function):
    '''
    Binarize the input activations and calculate the mean across channel dimension.
    '''
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self.save_for_backward(input)
        size = input.size()
        mean = torch.mean(input.abs(), 1, keepdim=True)
        input = input.sign()
        return input, mean

    def backward(
        self,
        grad_output: torch.Tensor,
        grad_output_mean: torch.Tensor,
    ) -> torch.Tensor:
        input, = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0
        return grad_input

class BinConv2d(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int=-1,
        stride: int=-1,
        padding: int=-1,
        dropout: float=0,
    ):
        super(BinConv2d, self).__init__()
        self.layer_type = 'BinConv2d'
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dropout_ratio = dropout

        self.bn = nn.BatchNorm2d(input_channels, eps=1e-4, momentum=0.1, affine=True)
        if dropout!=0:
            self.dropout = nn.Dropout(dropout)
        self.conv = nn.Conv2d(input_channels, output_channels,
                kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn(x)
        x, mean = BinActive()(x)
        if self.dropout_ratio!=0:
            x = self.dropout(x)
        x = self.conv(x)
        x = self.relu(x)
        return x

class BinConv1d(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int=-1,
        stride: int=1,
        padding: int=-1,
        dropout: float=0,
    ):
        super(BinConv1d, self).__init__()
        self.layer_type = 'BinConv1d'
        self.stride = stride
        self.dropout_ratio = dropout

        if dropout!=0:
            self.dropout = nn.Dropout(dropout)
        self.conv = nn.Conv1d(input_channels, output_channels,
                kernel_size=kernel_size, stride=stride, padding=padding)
        self.kernel_size = self.conv.kernel_size
        self.padding = self.conv.padding
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, mean = BinActive()(x)
        if self.dropout_ratio!=0:
            x = self.dropout(x)
        x = self.conv(x)
        x = self.relu(x)
        return x