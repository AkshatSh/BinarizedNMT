import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

def get_model_size(model, bits=64):
    """
    Return size in bits of a module.
    """
    mods = list(model.modules())
    sizes = []
    for i in range(len(mods)):
        m = mods[i]
        #TODO: Determine if isinstance(m, nn.Conv1d) is true
        p = list(m.parameters())
        for j in range(len(p)):
            sizes.append(np.array(p[j].size()))
    
    total_bits = 0
    for i in range(len(sizes)):
        s = sizes[i]
        bits = np.prod(np.array(s))*bits
        total_bits += bits

    return total_bits
