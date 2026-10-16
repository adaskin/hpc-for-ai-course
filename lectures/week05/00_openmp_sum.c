// Week 5 – OpenMP: shared memory, threads, one pragma.
// Build:  gcc -O2 -fopenmp 00_openmp_sum.c -o openmp_sum
// Run:    OMP_NUM_THREADS=1 ./openmp_sum ; OMP_NUM_THREADS=2 ./openmp_sum
#include <stdio.h>
#include <omp.h>

int main(void) {
    const long N = 200000000;
    double sum = 0.0;
    double t0 = omp_get_wtime();

    #pragma omp parallel for reduction(+:sum)
    for (long i = 0; i < N; i++)
        sum += 1.0 / (i + 1);

    printf("sum = %.10f  threads = %d  time = %.3f s\n",
           sum, omp_get_max_threads(), omp_get_wtime() - t0);
    return 0;
}
