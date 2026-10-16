// Week 5 – MPI: distributed memory, processes ("ranks"), explicit messages.
// Same problem as 00_openmp_sum.c, but every rank is a separate process
// with its own memory. Results are combined with an explicit MPI_Reduce.
// Build:  mpicc -O2 00_mpi_sum.c -o mpi_sum
// Run:    mpirun -n 4 ./mpi_sum        (add --oversubscribe if you have < 4 cores)
#include <stdio.h>
#include <mpi.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);   // who am I?
    MPI_Comm_size(MPI_COMM_WORLD, &size);   // how many of us?

    const long N = 200000000;
    double local = 0.0, total = 0.0;
    double t0 = MPI_Wtime();

    for (long i = rank; i < N; i += size)   // each rank takes its own share
        local += 1.0 / (i + 1);

    // Combine the partial sums on rank 0 (the "reduce" step).
    MPI_Reduce(&local, &total, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0)
        printf("sum = %.10f  ranks = %d  time = %.3f s\n",
               total, size, MPI_Wtime() - t0);

    MPI_Finalize();
    return 0;
}
