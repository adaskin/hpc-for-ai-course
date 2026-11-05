// 05_error_checking.cu -- how CUDA reports errors, and how to catch them.
//
// Three different kinds of failure:
//   1. an invalid launch configuration   (reported by cudaGetLastError)
//   2. a failed allocation               (reported by the return value of cudaMalloc)
//   3. a crash INSIDE the kernel         (reported later, by the next synchronizing call)
//
// Build & run:   nvcc -o 05_error_checking 05_error_checking.cu && ./05_error_checking
#include <cstdio>
#include <cstdlib>

// The macro you will paste into every CUDA program from now on:
// run the call, and stop with file + line + message if it failed.
#define CUDA_CHECK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) {            \
    fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
    exit(EXIT_FAILURE); } } while (0)

// For this demo only: print the result instead of exiting, so we can see all three.
static void report(const char *what, cudaError_t err) {
    printf("%-46s -> %s\n", what, err == cudaSuccess ? "OK" : cudaGetErrorString(err));
}

__global__ void write_one(int *p) { p[0] = 1; }

int main() {
    int *d_ok = nullptr;

    // A normal, successful call -- this is the CUDA_CHECK pattern you will use in real code.
    CUDA_CHECK(cudaMalloc(&d_ok, 4 * 1024 * 1024));
    printf("%-46s -> OK\n", "cudaMalloc 4 MB (CUDA_CHECK)");

    // A valid launch: nothing wrong, both checks say OK.
    write_one<<<1, 1>>>(d_ok);
    report("valid launch, cudaGetLastError", cudaGetLastError());
    report("valid launch, cudaDeviceSynchronize", cudaDeviceSynchronize());

    // 1. Invalid launch configuration: 2048 threads per block is above the 1024 limit.
    //    The launch statement itself returns nothing; the error is waiting in cudaGetLastError.
    write_one<<<1, 2048>>>(d_ok);
    report("bad launch (2048 threads/block)", cudaGetLastError());

    // 2. Allocation failure: ask for 1 TB. The error is in the return value.
    int *d_huge = nullptr;
    report("cudaMalloc 1 TB", cudaMalloc(&d_huge, 1ULL << 40));
    cudaGetLastError();               // clear the error we just handled, so it is not reported twice

    // 3. A crash inside the kernel: thread writes through a null pointer.
    //    The launch is accepted, the crash happens later, on the GPU.
    write_one<<<1, 1>>>(nullptr);
    report("null-pointer launch, cudaGetLastError", cudaGetLastError());
    report("null-pointer launch, cudaDeviceSynchronize", cudaDeviceSynchronize());

    // After an "illegal memory access" the CUDA context is broken ("sticky" error):
    // every later CUDA call fails too. The only fix is to end the process.
    report("cudaMalloc afterwards", cudaMalloc(&d_huge, 1024));
    return 0;
}
