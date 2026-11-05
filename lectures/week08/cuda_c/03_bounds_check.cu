// 03_bounds_check.cu -- why `if (i < n)` matters, made visible and safe.
//
// We want to fill n = 10 elements, but we launch 3 blocks x 4 threads = 12 threads.
// The buffer has 12 ints, so the two extra writes stay inside OUR allocation and
// cannot crash -- we can watch them overwrite two "guard" cells that hold -1.
// In real code those two writes would land in memory that is NOT yours.
//
// Build & run:   nvcc -o 03_bounds_check 03_bounds_check.cu && ./03_bounds_check
#include <cstdio>

__global__ void fill_safe(int *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = i;            // threads 10 and 11 skip the write
}

__global__ void fill_unsafe(int *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    out[i] = i;                       // BUG: no bounds check -> threads 10, 11 write too
}

void show(const char *label, const int *h) {
    printf("%-24s", label);
    for (int i = 0; i < 12; i++) {
        if (i == 10) printf(" |");    // everything right of "|" is beyond n
        printf(" %2d", h[i]);
    }
    printf("\n");
}

int main() {
    const int n = 10, total = 12;                    // 12 cells; only the first 10 are "ours"
    int h[total];
    int *d;
    cudaMalloc(&d, total * sizeof(int));

    // Run 1: with the bounds check
    cudaMemset(d, 0xFF, total * sizeof(int));        // every byte 0xFF -> every int is -1
    fill_safe<<<3, 4>>>(d, n);
    cudaMemcpy(h, d, total * sizeof(int), cudaMemcpyDeviceToHost);
    show("safe   (with i < n):", h);

    // Run 2: without the bounds check
    cudaMemset(d, 0xFF, total * sizeof(int));
    fill_unsafe<<<3, 4>>>(d, n);
    cudaMemcpy(h, d, total * sizeof(int), cudaMemcpyDeviceToHost);
    show("unsafe (no check):", h);

    cudaFree(d);
    return 0;
}
