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

#define THRESHOLD 42
#define CORRECTION 7

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

__global__ void accumulate(int *input, int *partial_sum) {
    int index = threadIdx.x;
    if (0 == index) *partial_sum = 0;
    __syncthreads();
    atomicAdd(partial_sum, input[index]);
}

__global__ void correct_accumulate(int *input, int *partial_sum, int *full_sum) {
    int index = threadIdx.x;
    if (*full_sum > THRESHOLD) {
        input[index] = input[index] - CORRECTION;
    }
    if (0 == index) *partial_sum = 0;
    __syncthreads();
    atomicAdd(partial_sum, input[index]);
}

int main(int c, char *v[]) {
    int mype, npes;
    int *input;
    int *partial_sum;
    int *full_sum;
    int input_nelems = 512;
    int to_all_nelems = 1;
    int PE_start = 0;
    int PE_size = 0;
    int logPE_stride = 0;
    long *pSync;
    int *pWrk;
    cudaStream_t stream;

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

    PE_size = nvshmem_n_pes();
    mype = nvshmem_my_pe();
    npes = nvshmem_n_pes();

    CUDA_CHECK(cudaSetDevice(mype));
    CUDA_CHECK(cudaStreamCreate(&stream));

    input = (int *)nvshmem_malloc(sizeof(int) * input_nelems);
    partial_sum = (int *)nvshmem_malloc(sizeof(int));
    full_sum = (int *)nvshmem_malloc(sizeof(int));
    pWrk = (int *)nvshmem_malloc(sizeof(int) * NVSHMEM_REDUCE_MIN_WRKDATA_SIZE);
    pSync = (long *)nvshmem_malloc(sizeof(long) * NVSHMEM_REDUCE_SYNC_SIZE);

    accumulate<<<1, input_nelems, 0, stream>>>(input, partial_sum);
    nvshmemx_int_sum_to_all_on_stream(full_sum, partial_sum, to_all_nelems, PE_start, logPE_stride,
                                      PE_size, pWrk, pSync, stream);
    correct_accumulate<<<1, input_nelems, 0, stream>>>(input, partial_sum, full_sum);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    printf("[%d of %d] run complete \n", mype, npes);

    CUDA_CHECK(cudaStreamDestroy(stream));

    nvshmem_free(input);
    nvshmem_free(partial_sum);
    nvshmem_free(full_sum);
    nvshmem_free(pWrk);
    nvshmem_free(pSync);

    nvshmem_finalize();

#ifdef ENABLE_MPI_SUPPORT
    if (use_mpi) MPI_Finalize();
#endif
    return 0;
}
