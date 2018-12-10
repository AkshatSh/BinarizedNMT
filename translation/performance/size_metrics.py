import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

def get_model_size(
    model: nn.Module,
    bits: int=64,
    binarized: bool=False,
) -> int:
    """
    Return size in bits of a module.
    """
    # mods = list(model.modules())
    sizes = []
    p = list(model.parameters())
    for j in range(len(p)):
        sizes.append(np.array(p[j].size()))
    
    binary_sizes = []
    alpha_sizes = []
    total_conv1d_params = []
    for m in model.modules():
        if isinstance(m, nn.Conv1d):
            binary_sizes.append(
                np.array(m.weight.data.size())
            )
            alpha_sizes.append(
                np.array((m.weight.data.size(0)))
            )
    
    total_conv1d_numbers = sum([np.prod(s) for s in total_conv1d_params])
    alpha_numbers = sum([np.prod(s) for s in alpha_sizes])
    total_sizes = sum([np.prod(s) for s in sizes])
    binary_numbers = sum([np.prod(s) for s in binary_sizes])

    if binarized:
        total_bits = (total_sizes - binary_numbers + alpha_numbers) * bits + binary_numbers
    else:
        total_bits = total_sizes * bits

    return total_bits
