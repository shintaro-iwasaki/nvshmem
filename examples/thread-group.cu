/*
 * Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 * See COPYRIGHT.txt for license information
 */

#include <stdio.h>
#include "nvshmem.h"
#include "nvshmemx.h"

#ifdef ENABLE_MPI_SUPPORT
#include "mpi.h"
#endif

#define NTHREADS 512

#undef CUDA_CHECK
#define CUDA_CHECK(stmt)                                                          \
    do {                                                                          \
        cudaError_t result = (stmt);                                              \
        if (cudaSuccess != result) {                                              \
            fprintf(stderr, "[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, \
                    cudaGetErrorString(result));                                  \
            exit(-1);                                                             \
        }                                                                         \
    } while (0)

__global__ void distributed_vector_sum(int *x, int *y, int *partial_sum, int *sum, long *pSync,
                                       int use_threadgroup, int mype, int npes) {
    int index = threadIdx.x;
    int nelems = blockDim.x;
    int PE_start = 0;
    int logPE_stride = 0;
    partial_sum[index] = x[index] + y[index];

    if (use_threadgroup) {
        /* all threads realize the entire collect operation */
        nvshmemx_collect32_block(sum, partial_sum, nelems, PE_start, logPE_stride, npes, pSync);
    } else {
        /* thread 0 realizes the entire collect operation */
        if (0 == index) {
            nvshmem_collect32(sum, partial_sum, nelems, PE_start, logPE_stride, npes, pSync);
        }
    }
}

int main(int c, char *v[]) {
    int mype, npes;
    int *x;
    int *y;
    int *partial_sum;
    int *sum;
    int use_threadgroup = 1;
    long *pSync;
    int nthreads = NTHREADS;

#ifdef ENABLE_MPI_SUPPORT
    bool use_mpi = false;
    char *value = getenv("NVSHMEMTEST_USE_MPI_LAUNCHER");
    if (value) use_mpi = atoi(value);
#endif

#ifdef ENABLE_MPI_SUPPORT
    if (use_mpi) {
        MPI_Init(&c, &v);
        int rank, nranks;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &nranks);
        MPI_Comm mpi_comm = MPI_COMM_WORLD;

        nvshmemx_init_attr_t attr;
        attr.mpi_comm = &mpi_comm;
        nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
    } else
        nvshmem_init();
#else
    nvshmem_init();
#endif

    npes = nvshmem_n_pes();
    mype = nvshmem_my_pe();

    CUDA_CHECK(cudaSetDevice(mype));

    x = (int *)nvshmem_malloc(sizeof(int) * nthreads);
    y = (int *)nvshmem_malloc(sizeof(int) * nthreads);
    partial_sum = (int *)nvshmem_malloc(sizeof(int) * nthreads);
    sum = (int *)nvshmem_malloc(sizeof(int) * nthreads * npes);
    pSync = (long *)nvshmem_malloc(sizeof(long) * NVSHMEM_COLLECT_SYNC_SIZE);

    void *args[] = {&x, &y, &partial_sum, &sum, &pSync, &use_threadgroup, &mype, &npes};
    dim3 dimBlock(nthreads);
    dim3 dimGrid(1);
    nvshmemx_collective_launch((const void *)distributed_vector_sum, dimGrid, dimBlock, args, 0, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("[%d of %d] run complete \n", mype, npes);

    nvshmem_free(x);
    nvshmem_free(y);
    nvshmem_free(partial_sum);
    nvshmem_free(sum);
    nvshmem_free(pSync);

    nvshmem_finalize();
#ifdef ENABLE_MPI_SUPPORT
    if (use_mpi) MPI_Finalize();
#endif

    return 0;
}
