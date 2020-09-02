/*
 * * Copyright (c) 2016-2018, NVIDIA CORPORATION. All rights reserved.
 * *
 * * See COPYRIGHT for license information
 * */

#include "nvshmem.h"
#include "nvshmemx_error.h"
#include "nvshmem_internal.h"
#include "transport.h"
#include "util.h"
#include "topo.h"

int nvshmemi_build_transport_map(nvshmem_state_t *state) {
    int status = 0;
    int *local_map = NULL;

    state->transport_map = (int *)calloc(state->npes * state->npes, sizeof(int));
    NULL_ERROR_JMP(state->transport_map, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                   "access map allocation failed \n");

    local_map = (int *)calloc(state->npes, sizeof(int));
    NULL_ERROR_JMP(local_map, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                   "access map allocation failed \n");

    state->transport_bitmap = 0;

    for (int i = 0; i < state->npes; i++) {
        int reach_any = 0;

        for (int j = 0; j < NVSHMEM_TRANSPORT_COUNT; j++) {
            int reach = 0;

            if (!state->transports[j]) {
                continue;
            }

            status = state->transports[j]->host_ops.can_reach_peer(&reach, &state->pe_info[i],
                                                                   state->transports[j]);
            NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "can reach peer failed \n");
            INFO(NVSHMEM_TOPO, "[%d] reach %d to peer %d over transport %d", state->mype, reach,
                 i, j);

            state->transports[j]->cap[i] = reach;
            reach_any |= reach;

            if (reach) {
                int m = 1 << j;
                local_map[i] |= m;
                // increment transport count if it has been picked for the first time
                if ((state->transport_bitmap & m) == 0) {
                    state->transport_count++;
                    state->transport_bitmap |= m;
                }
            }
        }

        if ((!reach_any) && (!nvshmemi_options.BYPASS_ACCESSIBILITY_CHECK)) {
            status = NVSHMEMX_ERROR_NOT_SUPPORTED;
            fprintf(stderr, "%s:%d: [GPU %d] Peer GPU %d is not accessible, exiting ... \n",
                    __FILE__, __LINE__, state->mype, i);
            goto out;
        }
    }
    INFO(NVSHMEM_TOPO, "[%d] transport bitmap: %x", state->mype, state->transport_bitmap);

    status = state->boot_handle.allgather((void *)local_map, (void *)state->transport_map,
                                          sizeof(int) * state->npes, &state->boot_handle);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "allgather of ipc handles failed \n");

out:
    if (local_map) free(local_map);
    if (status) {
        if (state->transport_map) free(state->transport_map);
    }
    return status;
}


int nvshmemi_detect_same_device(nvshmem_state_t *state) {
    int status = 0;
    nvshmem_transport_pe_info_t my_info;

    my_info.pe = state->mype;
    status = nvshmemi_get_pcie_attrs(&my_info.pcie_id, state->cudevice);
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out, "getPcieAttrs failed \n");

    my_info.hostHash = getHostHash();

    state->pe_info =
        (nvshmem_transport_pe_info_t *)malloc(sizeof(nvshmem_transport_pe_info_t) * state->npes);
    NULL_ERROR_JMP(state->pe_info, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                   "topo init info allocation failed \n");

    status = state->boot_handle.allgather((void *)&my_info, (void *)state->pe_info,
                                          sizeof(nvshmem_transport_pe_info_t), &state->boot_handle);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "allgather of ipc handles failed \n");

    for (int i = 0; i < state->npes; i++) {
        (state->pe_info + i)->pe = i;
        if (i == state->mype) continue;

        status = (((state->pe_info + i)->hostHash == my_info.hostHash) &&
                  ((state->pe_info + i)->pcie_id.dev_id == my_info.pcie_id.dev_id) &&
                  ((state->pe_info + i)->pcie_id.bus_id == my_info.pcie_id.bus_id) &&
                  ((state->pe_info + i)->pcie_id.domain_id == my_info.pcie_id.domain_id));
        NZ_ERROR_JMP(status, NVSHMEMX_ERROR_NOT_SUPPORTED, out,
                     "two PEs sharing a GPU is not supported \n");
    }

out:
    if (status) {
        state->cucontext = NULL;
        if (!state->pe_info) free(state->pe_info);
    }
    return status;
}


static int get_cuda_path(int cuda_dev, char **path) {
    int status = NVSHMEMX_SUCCESS;
    CUresult cu_err;
    char bus_id[16];
    char pathname[MAXPATHSIZE];
    char *cuda_rpath;
    char bus_path[] = "/sys/class/pci_bus/0000:00/device";

    cu_err = cuDeviceGetPCIBusId(bus_id, 16, cuda_dev);
    if (cu_err != CUDA_SUCCESS) {
        ERROR_PRINT("cuDeviceGetPCIBusId failed with error: %d \n", cu_err);
        status = NVSHMEMX_ERROR_INTERNAL;
        goto out;
    }
    INFO(NVSHMEM_TOPO, "[%d] user selected gpu with busid: %s ", nvshmem_state->mype, bus_id);

    for (int i = 0; i < 16; i++) bus_id[i] = tolower(bus_id[i]);
    memcpy(bus_path + sizeof("/sys/class/pci_bus/") - 1, bus_id, sizeof("0000:00") - 1);

    cuda_rpath = realpath(bus_path, NULL);
    NULL_ERROR_JMP(cuda_rpath, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out, "realpath failed \n");

    strncpy(pathname, cuda_rpath, MAXPATHSIZE);
    strncpy(pathname + strlen(pathname), "/", MAXPATHSIZE - strlen(pathname));
    strncpy(pathname + strlen(pathname), bus_id, MAXPATHSIZE - strlen(pathname));
    free(cuda_rpath);

    *path = realpath(pathname, NULL);
    NULL_ERROR_JMP(*path, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out, "realpath failed \n");

out:
    return status;
}

static int get_pci_distance(char *cuda_path, char *mlx_path) {
    int score = 0;
    int depth = 0;
    int same = 1;
    int i;
    for (i = 0; i < strlen(cuda_path); i++) {
        if (cuda_path[i] != mlx_path[i]) same = 0;
        if (cuda_path[i] == '/') {
            depth++;
            if (same == 1) score++;
        }
    }
    if (score <= 3) {
        /* Split the former PATH_SOC distance into PATH_NODE and PATH_SYS based on numaId */
        int numaId1 = getNumaId(cuda_path);
        int numaId2 = getNumaId(mlx_path);
        return ((numaId1 == numaId2) ? PATH_NODE : PATH_SYS);
    }
    if (score == 4) return PATH_PHB;
    if (score == depth - 1) return PATH_PIX;
    return PATH_PXB;
}

int get_device_by_distance(int *device, nvshmem_state_t *state, struct nvshmem_transport *tcurr) {
    int status = NVSHMEMX_SUCCESS;
    int dev_id, min_distance;
    int ndev = 0;
    int *distance = NULL;

    status = tcurr->host_ops.get_device_count(&ndev, tcurr);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                 "transport devices (setup_connections) failed \n");

    distance = (int *)malloc(sizeof(int) * ndev);
    NULL_ERROR_JMP(state->pe_info, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                   "distance allocation failed \n");

    char *cuda_path;
    status = get_cuda_path(state->cudevice, &cuda_path);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "get cuda path failed \n");

    for (int i = 0; i < ndev; i++) {
        char *dev_path;

        status = tcurr->host_ops.get_pci_path(i, &dev_path, tcurr);
        NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "get device path failed \n");

        distance[i] = get_pci_distance(cuda_path, dev_path);

        free(dev_path);
    }

    free(cuda_path);

    dev_id = 0;
    for (int i = 1; i < ndev; i++) {
        if (distance[i] < distance[dev_id]) dev_id = i;
    }

    if (distance[dev_id] > PATH_PXB) {
        if (!state->mype) {
            WARN_PRINT(
                "IB HCA and GPU are not connected to a PCIe switch "
                "so IB performance can be limited depending on the CPU generation \n");
        }
    }
    *device = dev_id;

    free(distance);
out:
    return status;
}
