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

#ifndef _NVSHMEM_COLL_COMMON_H_
#define _NVSHMEM_COLL_COMMON_H_

#include "nvshmem_common.cuh"
#include <cuda_runtime.h>

#define NVSHMEMI_ASET int PE_Start, int logPE_stride, int PE_size, long *pSync

#endif /* _NVSHMEM_COLL_COMMON_H_ */
