// 02_vector_add_small.cu -- the complete CUDA workflow on a tiny problem.
// c[i] = a[i] + b[i] for N = 10 elements, using blocks of 4 threads.
//
// Build & run:   nvcc -o 02_vector_add_small 02_vector_add_small.cu && ./02_vector_add_small
// (Error checking is left out on purpose so the workflow is easy to read.
//  Example 05 adds it, and every later example uses it.)
#include <cstdio>
#include <cstdlib>

__global__ void vector_add(const float *a, const float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;   // 1. who am I?
    if (i < n) {                                     // 2. is there work for me?
        c[i] = a[i] + b[i];                          // 3. do my one piece of work
    }
}

int main() {
    const int N = 10;
    const size_t bytes = N * sizeof(float);          // 10 * 4 = 40 bytes

    // Step 1: allocate and fill the HOST (CPU) arrays
    float *h_a = (float *)malloc(bytes);
    float *h_b = (float *)malloc(bytes);
    float *h_c = (float *)malloc(bytes);
    for (int i = 0; i < N; i++) {
        h_a[i] = (float)i;            // 0, 1, 2, ...
        h_b[i] = 10.0f * i;           // 0, 10, 20, ...
    }

    // Step 2: allocate DEVICE (GPU) arrays
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // Step 3: copy the inputs host -> device
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // Step 4: choose the launch configuration and launch
    int threads = 4;
    int blocks  = (N + threads - 1) / threads;       // ceil(10 / 4) = 3
    printf("N = %d, threads/block = %d, blocks = %d, launched threads = %d, idle threads = %d\n",
           N, threads, blocks, blocks * threads, blocks * threads - N);
    vector_add<<<blocks, threads>>>(d_a, d_b, d_c, N);

    // Step 5: copy the result device -> host (this call waits for the kernel to finish)
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    printf("c =");
    for (int i = 0; i < N; i++) printf(" %.0f", h_c[i]);
    printf("\n");

    // Step 6: free everything we allocated
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a);     free(h_b);     free(h_c);
    return 0;
}
