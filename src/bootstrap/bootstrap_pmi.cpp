/*
 * * Copyright (c) 2016-2018, NVIDIA CORPORATION. All rights reserved.
 * *
 * * See COPYRIGHT for license information
 * */

#include "nvshmem.h"

#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include "nvshmemx_error.h"
#include "util.h"
#include "pmi_internal.h"
#include "bootstrap.h"
#include "bootstrap_internal.h"

typedef struct {
    int singleton;
    int max_key_length;
    int max_value_length;
    int max_value_input_length;
    char *kvs_name;
    char *kvs_key;
    char *kvs_value;
} pmi_info_t;

static char encoding_table[] = {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                                'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
                                'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
                                'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
                                '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '/'};
static char *decoding_table = NULL;
static int mod_table[] = {0, 2, 1};

void base64_build_decoding_table() {
    decoding_table = (char *)malloc(256);

    for (int i = 0; i < 64; i++) decoding_table[(unsigned char)encoding_table[i]] = i;
}

void base64_cleanup() { free(decoding_table); }

size_t base64_encode_length(size_t in_len) { return (4 * ((in_len + 2) / 3)); }

size_t base64_decode_length(size_t in_len) { return (in_len / 4 * 3); }

size_t base64_encode(char *out, const unsigned char *in, size_t in_len) {
    size_t len = 4 * ((in_len + 2) / 3);

    for (int i = 0, j = 0; i < in_len;) {
        uint32_t a = i < in_len ? (unsigned char)in[i++] : 0;
        uint32_t b = i < in_len ? (unsigned char)in[i++] : 0;
        uint32_t c = i < in_len ? (unsigned char)in[i++] : 0;

        uint32_t fused = (a << 0x10) + (b << 0x08) + c;

        out[j++] = encoding_table[(fused >> 3 * 6) & 0x3F];
        out[j++] = encoding_table[(fused >> 2 * 6) & 0x3F];
        out[j++] = encoding_table[(fused >> 1 * 6) & 0x3F];
        out[j++] = encoding_table[(fused >> 0 * 6) & 0x3F];
    }

    for (int i = 0; i < mod_table[in_len % 3]; i++) out[len - 1 - i] = '=';

    return len;
}

size_t base64_decode(char *out, const char *in, size_t in_len) {
    size_t len = in_len / 4 * 3;

    if (in[in_len - 1] == '=') (len)--;
    if (in[in_len - 2] == '=') (len)--;

    for (int i = 0, j = 0; i < in_len;) {
        uint32_t a = in[i] == '=' ? 0 & i++ : decoding_table[in[i++]];
        uint32_t b = in[i] == '=' ? 0 & i++ : decoding_table[in[i++]];
        uint32_t c = in[i] == '=' ? 0 & i++ : decoding_table[in[i++]];
        uint32_t d = in[i] == '=' ? 0 & i++ : decoding_table[in[i++]];

        uint32_t fused = (a << 3 * 6) + (b << 2 * 6) + (c << 1 * 6) + (d << 0 * 6);

        if (j < len) out[j++] = (fused >> 2 * 8) & 0xFF;
        if (j < len) out[j++] = (fused >> 1 * 8) & 0xFF;
        if (j < len) out[j++] = (fused >> 0 * 8) & 0xFF;
    }

    return len;
}

int bootstrap_pmi_barrier(bootstrap_handle_t *handle) {
    int status = 0;

    status = SPMI_Barrier();
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "SPMI_Barrier failed \n");

out:
    return status;
}

int bootstrap_pmi_allgather(const void *sendbuf, void *recvbuf, int length,
                            bootstrap_handle_t *handle) {
    int status = 0, length64;
    static int key_index = 1;
    pmi_info_t *pmi_info;
    void *kvs_value;

    if (handle->pg_size == 1) {
        memcpy(recvbuf, sendbuf, length);
        return 0;
    }

    pmi_info = (pmi_info_t *)handle->internal;

    // TODO: this can be worked around by breaking down the transfer into multiple messages
    int max_length = pmi_info->max_value_input_length;
    int num_transfers = ((length + (max_length - 1)) / max_length);

    INFO(NVSHMEM_BOOTSTRAP, "PMI allgather: transfer length: %d max input length: %d", length,
         max_length);

    int processed = 0;
    int transfer = 0;
    int nbytes = 0;
    while (processed < length) {
        int curr_length = ((length - processed) > max_length) ? max_length : (length - processed);

        snprintf(pmi_info->kvs_key, pmi_info->max_key_length, "BOOTSTRAP-%04x-%08x-%04x", key_index,
                 handle->pg_rank, transfer);

        length64 = base64_encode((char *)pmi_info->kvs_value,
                                 (const unsigned char *)sendbuf + processed, curr_length);
        pmi_info->kvs_value[length64] = '\0';

        status = SPMI_KVS_Put(pmi_info->kvs_name, pmi_info->kvs_key, pmi_info->kvs_value);
        NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "SPMI_KVS_Put failed \n");

        status = SPMI_KVS_Commit();
        NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "SPMI_KVS_Commit failed \n");

        status = SPMI_Barrier();
        NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "SPMI_Barrier failed \n");

        for (int i = 0; i < handle->pg_size; i++) {
            snprintf(pmi_info->kvs_key, pmi_info->max_key_length, "BOOTSTRAP-%04x-%08x-%04x",
                     key_index, i, transfer);

            // assumes that same length is passed by all the processes
            status =
                SPMI_KVS_Get(pmi_info->kvs_name, pmi_info->kvs_key, pmi_info->kvs_value, length64);
            NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "SPMI_KVS_Get failed \n");

            base64_decode((char *)recvbuf + length * i + processed, (char *)pmi_info->kvs_value,
                          length64);
        }

        processed += curr_length;
        transfer++;
    }
out:
    key_index++;
    return status;
}

int bootstrap_pmi_alltoall(const void *sendbuf, void *recvbuf, int length,
                            bootstrap_handle_t *handle) {
    int status = 0, length64;
    static int key_index = 1;
    pmi_info_t *pmi_info;
    void *kvs_value;

    if (handle->pg_size == 1) {
        memcpy(recvbuf, sendbuf, length);
        return 0;
    }

    pmi_info = (pmi_info_t *)handle->internal;

    // TODO: this can be worked around by breaking down the transfer into multiple messages
    int max_length = pmi_info->max_value_input_length;
    int num_transfers = ((length + (max_length - 1)) / max_length);

    INFO(NVSHMEM_BOOTSTRAP, "PMI alltoall: transfer length: %d max input length: %d", length,
         max_length);

    int processed = 0;
    int transfer = 0;
    while (processed < length) {
        int curr_length = ((length - processed) > max_length) ? max_length : (length - processed);
        
        for (int i = 0; i < handle->pg_size; i++){
            snprintf(pmi_info->kvs_key, pmi_info->max_key_length, "BOOTSTRAP-%04x-%08x-%08x-%04x", key_index,
                     handle->pg_rank, i, transfer);

            length64 = base64_encode((char *)pmi_info->kvs_value,
                                     (const unsigned char *)sendbuf + i * length + processed, curr_length);
            pmi_info->kvs_value[length64] = '\0';

            status = SPMI_KVS_Put(pmi_info->kvs_name, pmi_info->kvs_key, pmi_info->kvs_value);
            NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "SPMI_KVS_Put failed \n");
        }

        status = SPMI_KVS_Commit();
        NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "SPMI_KVS_Commit failed \n");

        status = SPMI_Barrier();
        NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "SPMI_Barrier failed \n");

        for (int i = 0; i < handle->pg_size; i++) {
            snprintf(pmi_info->kvs_key, pmi_info->max_key_length, "BOOTSTRAP-%04x-%08x-%08x-%04x",
                     key_index, i, handle->pg_rank, transfer);

            // assumes that same length is passed by all the processes
            status =
                SPMI_KVS_Get(pmi_info->kvs_name, pmi_info->kvs_key, pmi_info->kvs_value, length64);
            NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "SPMI_KVS_Get failed \n");

            base64_decode((char *)recvbuf + length * i + processed, (char *)pmi_info->kvs_value,
                          length64);
        }

        processed += curr_length;
        transfer++;
    }
out:
    key_index++;
    return status;
}

int bootstrap_pmi_init(bootstrap_handle_t *handle) {
    int status = 0;
    int spawned = 0;
    int rank, size, key_length, value_length, name_length;
    pmi_info_t *pmi_info = NULL;

    status = SPMI_Init(&spawned);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "SPMI_Init_failed failed \n");

    pmi_info = (pmi_info_t *)calloc(1, sizeof(pmi_info_t));
    NULL_ERROR_JMP(pmi_info, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                   "memory allocation for pmi_info failed \n");

    if (!spawned) {
        INFO(NVSHMEM_BOOTSTRAP, "taking singleton path");

        // singleton launch
        handle->pg_rank = 0;
        handle->pg_size = 1;
        pmi_info->singleton = 1;
        handle->allgather = bootstrap_pmi_allgather;
        handle->alltoall = bootstrap_pmi_alltoall;
        handle->barrier = bootstrap_pmi_barrier;
    } else {
        status = SPMI_Get_rank(&rank);
        NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, error, "SPMI_Get_rank failed \n");

        status = SPMI_Get_size(&size);
        NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, error, "SPMI_Get_size failed \n");

        handle->pg_rank = rank;
        handle->pg_size = size;
        handle->allgather = bootstrap_pmi_allgather;
        handle->alltoall = bootstrap_pmi_alltoall;
        handle->barrier = bootstrap_pmi_barrier;

        status = SPMI_KVS_Get_name_length_max(&name_length);
        NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, error,
                     "SPMI_KVS_Get_name_length_max failed \n");

        status = SPMI_KVS_Get_key_length_max(&key_length);
        NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, error,
                     "SPMI_KVS_Get_key_length_max failed \n");

        status = SPMI_KVS_Get_value_length_max(&value_length);
        NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, error,
                     "SPMI_KVS_Get_value_length_max failed \n");

        pmi_info->max_key_length = key_length;
        pmi_info->max_value_length = value_length;

        // hacky workaround to allow space for metadata in KVS_Put, needs investgation
        pmi_info->max_value_input_length = base64_decode_length(value_length / 2);
        INFO(NVSHMEM_BOOTSTRAP, "PMI max key length: %d max value length %d", key_length,
             value_length);

        pmi_info->kvs_name = (char *)malloc(name_length);
        pmi_info->kvs_key = (char *)malloc(key_length);
        pmi_info->kvs_value = (char *)malloc(value_length);

        NULL_ERROR_JMP(pmi_info->kvs_name, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                       "memory allocation for kvs_name failed \n");

        status = SPMI_KVS_Get_my_name(pmi_info->kvs_name, name_length);
        NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, error, "SPMI_KVS_Get_my_name failed \n");
    }

    handle->internal = (void *)pmi_info;

    base64_build_decoding_table();

error:
    if (status && pmi_info) {
        if (pmi_info->kvs_name) free(pmi_info->kvs_name);
        free(pmi_info);
    }
out:
    return status;
}

int bootstrap_pmi_finalize(bootstrap_handle_t *handle) {
    int status = 0;

    pmi_info_t *pmi_info = (pmi_info_t *)handle->internal;

    status = SPMI_Finalize();
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, error, "SPMI_KVS_Get_my_name failed \n");

    base64_cleanup();

error:
    if (pmi_info) {
        if (pmi_info->kvs_name) free(pmi_info->kvs_name);
        free(pmi_info);
    }

out:
    return status;
}
