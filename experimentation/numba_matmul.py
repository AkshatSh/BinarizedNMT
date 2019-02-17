from numba import cuda, float32
import torch

TPB = 16

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
    res_matrix = torch.zeros(size=(a.shape[0], b.shape[1]))
    print(res_matrix.shape)
    a_np, b_np, res_np = a.cpu().numpy(), b.cpu().numpy(), res_matrix.cpu().numpy()
    fast_matmul(a_np, b_np, res_np)
    return torch.Tensor(res_np)

#res = mat_mul(torch.Tensor([[1]]), torch.Tensor([[1]]))
#print(res)
