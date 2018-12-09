"""
test using torch unfold to do a convolution

we'll do a convolution both using standard conv, and unfolding it and matrix mul,
and try to get the same answer
"""
import torch
from torch import nn, optim
import torch.nn.functional as F

def run():
    in_channels = 256
    out_channels = 512
    size = 200
    torch.manual_seed(123)
    X = torch.rand(128, in_channels, size)
    conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False)
    conv.weight.data = conv.weight.data.sign()
    out = conv(X)
    # print('out', out)
    print('out.size()', out.size())
    print('')

    # Xunfold = F.unfold(X, kernel_size=3, padding=1)
    Xunfold1d = X.unfold(1, 3, 1)
    inp = X.unsqueeze(-1)
    inp_unf = torch.nn.functional.unfold(inp, (3, 1), padding=(1,0))
    print('inp_unf', inp_unf.shape)
    # print('X.size()', X.size())
    print(X.shape)
    print('Xunfold1d.size()', Xunfold1d.size())
    print(conv.weight.data.shape)
    kernels_flat = conv.weight.data.view(out_channels, -1)
    print('kernels_flat.size()', kernels_flat.size())

    print('({}) @ ({})'.format(kernels_flat.shape, inp_unf.shape))
    
    res = kernels_flat @ inp_unf
    # res = res.view(1, out_channels, size)
    # print('res', res)
    print('res.size()', res.size())
    print((out - res).sum())

run()