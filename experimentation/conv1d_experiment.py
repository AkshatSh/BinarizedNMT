"""
test using torch unfold to do a convolution

we'll do a convolution both using standard conv, and unfolding it and matrix mul,
and try to get the same answer
"""
import torch
from torch import nn, optim
import torch.nn.functional as F

# from numba_matmul import mat_mul

def bin_matmul(a, b):
    '''
    a is represented as eitehr -1 or 1
    '''
    output = torch.Tensor(a.shape[0], b.shape[1])
    for row_i in range(a.shape[0]):
        for internal in range(a.shape[1]):
            for col_i in range(b.shape[1]):
                curr_a = a[row_i][internal]
                curr_b = b[internal][col_i]
                term = curr_b if curr_a > 0 else -curr_b
                output[row_i][col_i] += term
    
    return output

def matmul(a, b):
    output = torch.Tensor(a.shape[0], b.shape[1])
    for row_i in range(a.shape[0]):
        for internal in range(a.shape[1]):
            for col_i in range(b.shape[1]):
                curr_a = a[row_i][internal]
                curr_b = b[internal][col_i]
                output[row_i][col_i] += a * b
    
    return output

def run():
    batch = 1
    in_channels = 25
    out_channels = 51
    size = 20
    torch.manual_seed(123)
    X = torch.rand(batch, in_channels, size)
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


    output = torch.Tensor(
        batch,
        out_channels,
        size,
    )

    output_numba = torch.Tensor(
        batch,
        out_channels,
        size,
    )

    
    # kernels_flat is only -1 or 1 
    for batch_i in range(batch):
        # multiply kernel_flat by inp_unf[batch_i]
        curr_matrix = inp_unf[batch_i]
        output[batch_i] = bin_matmul(kernels_flat, curr_matrix)
        # output_numba[batch_i] = mat_mul(kernels_flat, curr_matrix)

    # res = kernels_flat @ inp_unf
    res = output
    # res = res.view(1, out_channels, size)
    # print('res', res)
    print('res.size()', res.size())
    print((out - res).sum())

run()
