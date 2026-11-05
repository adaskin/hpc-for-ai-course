// 07_grid_stride.cu -- one fixed-size grid that handles ANY array length.
//
//   Part A: who handles which element?  (N = 20, only 8 threads)
//   Part B: SAXPY  y = a*x + y  with the same launch for three different N
//
// Build & run:   nvcc -o 07_grid_stride 07_grid_stride.cu && ./07_grid_stride
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) {            \
    fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
    exit(EXIT_FAILURE); } } while (0)

// Part A: record which thread processed element i
__global__ void who_did_it(int *owner, int n) {
    int first  = blockIdx.x * blockDim.x + threadIdx.x;   // my first element = my global id
    int stride = blockDim.x * gridDim.x;                  // total number of threads launched
    for (int i = first; i < n; i += stride) {
        owner[i] = first;
    }
}

// Part B: y = a * x + y, grid-stride version
__global__ void saxpy(float a, const float *x, float *y, int n) {
    int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        y[i] = a * x[i] + y[i];
    }
}

int main() {
    // ---- Part A --------------------------------------------------------------
    const int n = 20;
    int *d_owner, h_owner[n];
    CUDA_CHECK(cudaMalloc(&d_owner, n * sizeof(int)));
    who_did_it<<<2, 4>>>(d_owner, n);                     // 2 blocks x 4 threads = 8 threads
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_owner, d_owner, n * sizeof(int), cudaMemcpyDeviceToHost));

    printf("Part A: N = %d, 8 threads, stride = 8\n", n);
    printf("element:");
    for (int i = 0; i < n; i++) printf(" %2d", i);
    printf("\nthread :");
    for (int i = 0; i < n; i++) printf(" %2d", h_owner[i]);
    printf("\n");
    CUDA_CHECK(cudaFree(d_owner));

    // ---- Part B --------------------------------------------------------------
    printf("\nPart B: saxpy, always 32 blocks x 256 threads = 8192 threads\n");
    int sizes[] = {1000, 1 << 20, 5000000};
    for (int k = 0; k < 3; k++) {
        int N = sizes[k];
        size_t bytes = N * sizeof(float);
        float *h_y = (float *)malloc(bytes);
        float *d_x, *d_y;
        CUDA_CHECK(cudaMalloc(&d_x, bytes));
        CUDA_CHECK(cudaMalloc(&d_y, bytes));

        // x = 1 everywhere, y = 2 everywhere  ->  y = 2*1 + 2 = 4 everywhere
        float *h_x = (float *)malloc(bytes);
        for (int i = 0; i < N; i++) { h_x[i] = 1.0f; h_y[i] = 2.0f; }
        CUDA_CHECK(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_y, h_y, bytes, cudaMemcpyHostToDevice));

        saxpy<<<32, 256>>>(2.0f, d_x, d_y, N);            // same launch for every N
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(h_y, d_y, bytes, cudaMemcpyDeviceToHost));

        int wrong = 0;
        for (int i = 0; i < N; i++) if (h_y[i] != 4.0f) wrong++;
        printf("N = %8d: elements per thread ~ %5.1f, wrong elements = %d\n",
               N, (double)N / (32 * 256), wrong);

        CUDA_CHECK(cudaFree(d_x)); CUDA_CHECK(cudaFree(d_y));
        free(h_x); free(h_y);
    }
    return 0;
}
