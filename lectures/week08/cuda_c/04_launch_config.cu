// 04_launch_config.cu -- choosing blocks and threads for N elements.
//
//   Part A: the ceiling-division formula, for many values of N (host only)
//   Part B: what the GPU allows (device properties)
//   Part C: the same N = 1,000,000 vector add with different block sizes,
//           and what happens if you round DOWN instead of UP
//
// Build & run:   nvcc -o 04_launch_config 04_launch_config.cu && ./04_launch_config
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) {            \
    fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
    exit(EXIT_FAILURE); } } while (0)

__global__ void vector_add(const float *a, const float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

// ceil(n / threads) using integer arithmetic only
int ceil_div(int n, int threads) { return (n + threads - 1) / threads; }

// Run the kernel with `blocks` x `threads`, return how many elements are wrong.
int run_and_count_wrong(int n, int blocks, int threads,
                        const float *d_a, const float *d_b, float *d_c, float *h_c) {
    CUDA_CHECK(cudaMemset(d_c, 0, n * sizeof(float)));           // start from all zeros
    vector_add<<<blocks, threads>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaGetLastError());                              // was the launch valid?
    CUDA_CHECK(cudaMemcpy(h_c, d_c, n * sizeof(float), cudaMemcpyDeviceToHost));
    int wrong = 0;
    for (int i = 0; i < n; i++)
        if (h_c[i] != 3.0f * i) wrong++;                         // expected: i + 2i = 3i
    return wrong;
}

int main() {
    // ---- Part A: the formula ------------------------------------------------
    printf("Part A: blocks = ceil(N / 256)\n");
    printf("%12s %8s %16s %6s\n", "N", "blocks", "launched threads", "idle");
    int ns[] = {1, 256, 257, 1000, 1000000, 10000000};
    for (int k = 0; k < 6; k++) {
        int n = ns[k], blocks = ceil_div(n, 256);
        printf("%12d %8d %16d %6d\n", n, blocks, blocks * 256, blocks * 256 - n);
    }

    // ---- Part B: hardware limits ---------------------------------------------
    cudaDeviceProp p;
    CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
    printf("\nPart B: device 0 = %s\n", p.name);
    printf("  max threads per block : %d\n", p.maxThreadsPerBlock);
    printf("  warp size             : %d\n", p.warpSize);
    printf("  streaming multiprocs  : %d\n", p.multiProcessorCount);

    // ---- Part C: same answer, different launch configurations ---------------
    const int N = 1000000;
    size_t bytes = N * sizeof(float);
    float *h_a = (float *)malloc(bytes), *h_b = (float *)malloc(bytes), *h_c = (float *)malloc(bytes);
    for (int i = 0; i < N; i++) { h_a[i] = (float)i; h_b[i] = 2.0f * i; }   // c should be 3i
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    printf("\nPart C: N = %d\n", N);
    printf("%10s %8s %16s %10s\n", "threads", "blocks", "launched threads", "wrong");
    int sizes[] = {32, 128, 256, 1024};
    for (int k = 0; k < 4; k++) {
        int t = sizes[k], b = ceil_div(N, t);
        int wrong = run_and_count_wrong(N, b, t, d_a, d_b, d_c, h_c);
        printf("%10d %8d %16d %10d\n", t, b, b * t, wrong);
    }

    // The classic mistake: floor division leaves the tail uncomputed.
    int b_floor = N / 256;                                       // 3906, NOT 3907
    int wrong = run_and_count_wrong(N, b_floor, 256, d_a, d_b, d_c, h_c);
    printf("\nfloor division: blocks = %d, launched threads = %d, wrong = %d\n",
           b_floor, b_floor * 256, wrong);

    CUDA_CHECK(cudaFree(d_a)); CUDA_CHECK(cudaFree(d_b)); CUDA_CHECK(cudaFree(d_c));
    free(h_a); free(h_b); free(h_c);
    return 0;
}
