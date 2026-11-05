// 01_hello_indices.cu -- "Who am I?" Every thread prints its own coordinates.
//
// Build & run:   nvcc -o 01_hello_indices 01_hello_indices.cu && ./01_hello_indices
#include <cstdio>

__global__ void hello_indices() {
    // The core formula: block number * threads per block + thread number in block
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    printf("block %d, thread %d  ->  i = %d * %d + %d = %2d\n",
           blockIdx.x, threadIdx.x, blockIdx.x, blockDim.x, threadIdx.x, i);
}

int main() {
    // <<<blocks, threads_per_block>>> : 3 blocks x 4 threads = 12 threads in total
    hello_indices<<<3, 4>>>();

    // Kernel launches are asynchronous: wait here, otherwise main() may exit
    // before the GPU has printed anything.
    cudaDeviceSynchronize();
    return 0;
}
