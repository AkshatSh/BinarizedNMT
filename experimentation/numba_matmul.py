from numba import cuda, float32, guvectorize, void, float64
import torch
import math

@guvectorize([void(float64[:,:], float64[:,:], float64[:,:])], '(m,l),(l,n)->(m,n)', target='cuda')
def matmul_gu3(A, B, out):
    """Perform square matrix multiplication of out = A * B
    """
    i, j = cuda.grid(2)
    if i < out.shape[0] and j < out.shape[1]:
        tmp = 0.
        for k in range(A.shape[1]):
            tmp += A[i, k] * B[k, j]
        out[i, j] = tmp
 
matmul_gu3.max_blocksize = 32

TPB = 16
threadsperblock = (TPB, TPB)


@cuda.jit
def fast_matmul(A, B, C):
    # Define an array in the shared memory
    # The size and type of the arrays must be known at compile time
    sA = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
    sB = cuda.shared.array(shape=(TPB, TPB), dtype=float32)

    x, y = cuda.grid(2)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bpg = cuda.gridDim.x    # blocks per grid

    if x >= C.shape[0] and y >= C.shape[1]:
        # Quit if (x, y) is outside of valid C boundary
        return

    # Each thread computes one element in the result matrix.
    # The dot product is chunked into dot products of TPB-long vectors.
    tmp = 0.
    for i in range(bpg):
        # Preload data into shared memory
        sA[tx, ty] = A[x, ty + i * TPB]
        sB[tx, ty] = B[tx + i * TPB, y]

        # Wait until all threads finish preloading
        cuda.syncthreads()

        # Computes partial product on the shared memory
        for j in range(TPB):
            tmp += sA[tx, j] * sB[j, ty]

        # Wait until all threads finish computing
        cuda.syncthreads()

    C[x, y] = tmp

def mat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    stream = cuda.stream()
    res_matrix = torch.zeros(size=(a.shape[0], b.shape[1]))
    blockspergrid_x = int(math.ceil(res_matrix.shape[0] / threadsperblock[0]))
    blockspergrid_y = int(math.ceil(res_matrix.shape[1] / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    print(res_matrix.shape)

    a_np, b_np, res_np = (
        a.cpu().numpy(),
        b.cpu().numpy(),
        res_matrix.cpu().numpy(),
    )

    stream.synchronize()
    d_a = cuda.to_device(a_np, stream=stream)
    d_b = cuda.to_device(b_np, stream=stream)
    # stream.synchronize()

    fast_matmul[blockspergrid, threadsperblock, stream](a_np, b_np, res_np)
    return torch.Tensor(res_np)

#res = mat_mul(torch.Tensor([[1]]), torch.Tensor([[1]]))
#print(res)
