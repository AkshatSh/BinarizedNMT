
import torch.nn as nn
import torch
import torch.nn.functional as F

from .binarized_utils import BinActive

class BinLinear(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        is_xnor: bool = False,
    ):
        super(BinLinear, self).__init__()
        self.layer_type = 'BinLinear'
        self.is_xnor = is_xnor

        self.linear = nn.Linear(input_channels, output_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_xnor:
            x, mean = BinActive()(x)
        x = self.linear(x)
        # x = self.relu(x)
        return 

class XNORLinear(BinLinear):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
    ):
        BinLinear.__init__(
            self,
            input_channels=input_channels,
            output_channels=output_channels,
            is_xnor=True,
        )

        self.layer_type = 'XNORLinear'