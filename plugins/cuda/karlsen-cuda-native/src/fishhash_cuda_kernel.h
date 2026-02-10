#pragma once

#include <stdint.h>

typedef union {
    uint64_t word64s[4];
    uint32_t word32s[8];
    uint8_t bytes[32];
    char str[32];
    uint2 uint2s[4];
} hash256;

typedef union {
    uint64_t word64s[8];
    uint32_t word32s[16];
    uint8_t bytes[64];
    char str[64];
    uint2 uint2s[8];
} hash512;

typedef union {
    hash512 hash512s[2];
    uint64_t word64s[16];
    uint32_t word32s[32];
    uint8_t bytes[128];
    char str[128];
    uint2 uint2s[16];
} hash1024;

typedef struct {
    const int light_cache_num_items;
    const hash512* light_cache;
    const int full_dataset_num_items;
    hash1024* full_dataset; // mutable for lazy loading?
} fishhash_context;


#define CUDA_SAFE_CALL(call)                                                              \
    do                                                                                    \
    {                                                                                     \
        cudaError_t err = call;                                                           \
        if (cudaSuccess != err)                                                           \
        {                                                                                 \
            std::stringstream ss;                                                         \
            ss << "CUDA error in func " << __FUNCTION__ << " at line " << __LINE__ << ' ' \
               << cudaGetErrorString(err);                                                \
            throw cuda_runtime_error(ss.str());                                           \
        }                                                                                 \
    } while (0)
