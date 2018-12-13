'''
This is from the Pytorch implementation of XNOR-NET

Implementation is linked here: https://github.com/jiecaoyu/XNOR-Net-PyTorch/blob/master/CIFAR_10/models/nin.py
'''

import torch.nn as nn
import torch
import torch.nn.functional as F

from .binarized_utils import BinActive

USE_SIMPLE = False

class BinConv1d(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int=-1,
        stride: int=1,
        padding: int=-1,
        dropout: float=0,
        is_xnor: bool = False,
    ):
        super(BinConv1d, self).__init__()
        self.layer_type = 'BinConv1d'
        self.stride = stride
        self.dropout_ratio = dropout
        self.is_xnor = is_xnor

        if dropout!=0:
            self.dropout = nn.Dropout(dropout)
        self.conv = nn.Conv1d(input_channels, output_channels,
                kernel_size=kernel_size, stride=stride, padding=padding)
        self.kernel_size = self.conv.kernel_size
        self.padding = self.conv.padding
        # self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_xnor:
            x, mean = BinActive()(x)
        if self.dropout_ratio!=0:
            x = self.dropout(x)
        x = self.conv(x)
        # x = self.relu(x)
        return x

class BinConvTBC(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int=-1,
        stride: int=1,
        padding: int=-1,
        dropout: float=0,
    ):
        super(BinConvTBC, self).__init__()
        '''
        Implementation of binary convolution using TBC (time batch channel)
        as compared to 1d convolutions using BCT (batch channel time)
        '''
        self.layer_type = 'BinConvTBC'
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _single(kernel_size)
        self.padding = _single(padding)

        self.weight = torch.nn.Parameter(torch.Tensor(
            self.kernel_size[0], in_channels, out_channels))
        self.bias = torch.nn.Parameter(torch.Tensor(out_channels))

        if not USE_SIMPLE:
            self.special_init()
    
    def special_init(self):
        nn.init.normal_(self.weight, mean=0, std=math.sqrt((1 - dropout) / in_features))
        nn.init.constant_(self.bias, 0)
        return nn.utils.weight_norm(self)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, mean = BinActive()(x)
        return torch.conv_tbc(x.contiguous(), self.weight, self.bias, self.padding[0])
    
    def __repr__(self):
        s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', padding={padding}')
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)

class XNORConv1d(BinConv1d):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int=-1,
        stride: int=1,
        padding: int=-1,
        dropout: float=0,
    ):
        super(BinConv1d, self).__init__(
            input_channels=input_channels,
            output_channels=output_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dropout=dropout,
            is_xnor=True,
        )

        self.layer_type = 'XNORConv1d'