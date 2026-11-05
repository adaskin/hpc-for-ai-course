// 06_matrix_2d.cu -- 2D thread indexing on a 5 x 7 matrix.
//
// Problem: out[row][col] = 10 * row + col, one thread per matrix element.
// A matrix is stored as ONE flat array in row-major order, so element (row, col)
// lives at position  row * n_cols + col.
//
// Build & run:   nvcc -o 06_matrix_2d 06_matrix_2d.cu && ./06_matrix_2d
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) {            \
    fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
    exit(EXIT_FAILURE); } } while (0)

__global__ void fill_matrix(int *out, int n_rows, int n_cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;   // x -> columns
    int row = blockIdx.y * blockDim.y + threadIdx.y;   // y -> rows

    if (row < n_rows && col < n_cols) {                // check BOTH dimensions
        out[row * n_cols + col] = 10 * row + col;      // flat row-major position
    }
}

int main() {
    const int n_rows = 5, n_cols = 7;
    const size_t bytes = n_rows * n_cols * sizeof(int);

    int *d_out;
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemset(d_out, 0xFF, bytes));        // fill with -1 so untouched cells would show

    dim3 threads(4, 4);                                // (x, y) = 4 columns x 4 rows = 16 threads
    dim3 blocks((n_cols + threads.x - 1) / threads.x,  // ceil(7 / 4) = 2 blocks across
                (n_rows + threads.y - 1) / threads.y); // ceil(5 / 4) = 2 blocks down
    printf("grid = (%d, %d) blocks, block = (%d, %d) threads -> %d x %d = %d threads, %d do work\n",
           blocks.x, blocks.y, threads.x, threads.y,
           blocks.x * threads.x, blocks.y * threads.y,
           blocks.x * threads.x * blocks.y * threads.y, n_rows * n_cols);

    fill_matrix<<<blocks, threads>>>(d_out, n_rows, n_cols);
    CUDA_CHECK(cudaGetLastError());

    int h_out[n_rows * n_cols];
    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

    for (int r = 0; r < n_rows; r++) {
        for (int c = 0; c < n_cols; c++) printf("%3d", h_out[r * n_cols + c]);
        printf("\n");
    }

    CUDA_CHECK(cudaFree(d_out));
    return 0;
}
