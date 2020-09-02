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

#ifndef _NVSHMEM_CONSTANTS_H_
#define _NVSHMEM_CONSTANTS_H_

#define SYNC_SIZE 27648 /*XXX:Number of GPUs on Summit; currently O(N), need to be O(1)*/

#define NVSHMEM_SYNC_VALUE 0
#define NVSHMEM_SYNC_SIZE (2 * SYNC_SIZE)
#define NVSHMEM_BARRIER_SYNC_SIZE (2 * SYNC_SIZE)
#define NVSHMEM_BCAST_SYNC_SIZE SYNC_SIZE
#define NVSHMEM_REDUCE_SYNC_SIZE SYNC_SIZE
#define NVSHMEM_REDUCE_MIN_WRKDATA_SIZE SYNC_SIZE
#define NVSHMEM_COLLECT_SYNC_SIZE SYNC_SIZE
#define NVSHMEM_ALLTOALL_SYNC_SIZE SYNC_SIZE

#define _NVSHMEM_SYNC_VALUE NVSHMEM_SYNC_VALUE
#define _NVSHMEM_BARRIER_SYNC_SIZE NVSHMEM_BARRIER_SYNC_SIZE
#define _NVSHMEM_BCAST_SYNC_SIZE NVSHMEM_BCAST_SYNC_SIZE
#define _NVSHMEM_REDUCE_SYNC_SIZE NVSHMEM_REDUCE_SYNC_SIZE
#define _NVSHMEM_REDUCE_MIN_WRKDATA_SIZE NVSHMEM_REDUCE_MIN_WRKDATA_SIZE
#define _NVSHMEM_COLLECT_SYNC_SIZE NVSHMEM_COLLECT_SYNC_SIZE
#define _NVSHMEM_ALLTOALL_SYNC_SIZE NVSHMEM_ALLTOALL_SYNC_SIZE

#define NVSHMEM_MAJOR_VERSION 1
#define NVSHMEM_MINOR_VERSION 3
#define _NVSHMEM_MAJOR_VERSION NVSHMEM_MAJOR_VERSION
#define _NVSHMEM_MINOR_VERSION NVSHMEM_MINOR_VERSION

#define NVSHMEM_VENDOR_STRING "NVSHMEM v1.0.1"
#define _NVSHMEM_VENDOR_STRING NVSHMEM_VENDOR_STRING

#define NVSHMEM_MAX_NAME_LEN 256
#define _NVSHMEM_MAX_NAME_LEN NVSHMEM_MAX_NAME_LEN

enum nvshmemi_cmp_type {
    NVSHMEM_CMP_EQ = 0,
    NVSHMEM_CMP_NE,
    NVSHMEM_CMP_GT,
    NVSHMEM_CMP_LE,
    NVSHMEM_CMP_LT,
    NVSHMEM_CMP_GE
};

enum nvshmemi_thread_support {
    NVSHMEM_THREAD_SINGLE = 0,
    NVSHMEM_THREAD_FUNNELED,
    NVSHMEM_THREAD_SERIALIZED,
    NVSHMEM_THREAD_MULTIPLE
};

enum nvshmemi_predefined_teams {
    NVSHMEM_TEAM_INVALID = 0,
    NVSHMEM_TEAM_WORLD,
    NVSHMEMX_TEAM_NODE
};

#endif
