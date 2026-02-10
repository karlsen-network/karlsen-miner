#include<stdint.h>
#include <assert.h>
#include "keccak-tiny.c"
#include "xoshiro256starstar.c"
#include "fishhash_cuda_kernel.cuh"


typedef uint8_t Hash[32];

typedef union _uint256_t {
    uint64_t number[4];
    uint8_t hash[32];
} uint256_t;

#define BLOCKDIM 1024
#define HASH_HEADER_SIZE 72
#define LIGHT_CACHE_NUM_ITEMS 1179641

#define RANDOM_LEAN 0
#define RANDOM_XOSHIRO 1

DEV_INLINE void keccak_in_place(uint8_t* data) {
    SHA3_512((uint2*)data);
}

#define LT_U256(X,Y) (X.number[3] != Y.number[3] ? X.number[3] < Y.number[3] : X.number[2] != Y.number[2] ? X.number[2] < Y.number[2] : X.number[1] != Y.number[1] ? X.number[1] < Y.number[1] : X.number[0] < Y.number[0])

__constant__ uint8_t hash_header[HASH_HEADER_SIZE];
__constant__ uint256_t target;

extern "C" __global__ void generate_full_dataset_gpu(
    int light_cache_num_items, // 1179641
    hash512* light_cache,
    int full_dataset_num_items, // 37748717
    hash1024* full_dataset
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index % 1000000 == 0 && threadIdx.x == 0) {
        printf("[GPU] Generating DAG item %d / %d\n", index, full_dataset_num_items);
    }
    if (index >= full_dataset_num_items) return;

    fishhash_context ctx = {light_cache_num_items, light_cache, full_dataset_num_items, full_dataset};
    full_dataset[index] = calculate_dataset_item_1024(ctx, index);
}

/*
extern "C" __global__ void build_light_cache_gpu(hash512* cache_out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= LIGHT_CACHE_NUM_ITEMS) return;

    __shared__ hash512 prev;

    if (i == 0) {
        uint8_t seed[32] = {
            0xeb, 0x01, 0x63, 0xae, 0xf2, 0xab, 0x1c, 0x5a, 0x66, 0x31, 0x0c, 0x1c, 0x14, 0xd6, 0x0f, 0x42,
            0x55, 0xa9, 0xb3, 0x9b, 0x0e, 0xdf, 0x26, 0x53, 0x98, 0x44, 0xf1, 0x17, 0xad, 0x67, 0x21, 0x19
        };
        memset(prev.bytes, 0, 64);
        memcpy(prev.bytes, seed, 32);
        keccak_in_place(prev.bytes);
        cache_out[0] = prev;
    } else {
        __syncthreads();
        hash512 item = cache_out[i - 1];
        keccak_in_place(item.bytes);
        cache_out[i] = item;
    }
}
*/

extern "C" {

    __global__ void khashv2_kernel(
            const uint64_t nonce_mask, 
            const uint64_t nonce_fixed, 
            const uint64_t nonces_len, 
            uint8_t random_type, 
            void* states, 
            uint64_t *final_nonce,
            hash1024* dataset,
            hash512* cache
            ) {

        // assuming header_len is 72
        /*
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            printf("khashv2_kernel Thread %d, Block %d\n", threadIdx.x, blockIdx.x);
            printHash("The cache[10] is : ", cache[10].bytes, 128);
            printHash("The cache[42] is : ", cache[42].bytes, 128);
            printHash("The dataset[10] is : ", dataset[10].bytes, 128);
            printHash("The dataset[42] is : ", dataset[42].bytes, 128);
            printHash("The dataset[12345] is : ", dataset[12345].bytes, 128);
        }
        */
        

        int nonceId = threadIdx.x + blockIdx.x*blockDim.x;
        if (nonceId < nonces_len) {
            if (nonceId == 0) *final_nonce = 0;
            uint64_t nonce;
            switch (random_type) {
                case RANDOM_LEAN:
                    nonce = ((uint64_t *)states)[0] ^ nonceId;
                    break;
                case RANDOM_XOSHIRO:
                default:
                    nonce = xoshiro256_next(((ulonglong4 *)states) + nonceId);
                    break;
            }
            nonce = (nonce & nonce_mask) | nonce_fixed;
            // header
            uint8_t input[80];
            memcpy(input, hash_header, HASH_HEADER_SIZE);
            // data
            // TODO: check endianity?
            uint256_t hash_;
            memcpy(input +  HASH_HEADER_SIZE, (uint8_t *)(&nonce), 8);
            hashB3(hash_.hash, input, 80);

            /*
            if (threadIdx.x == 0 && blockIdx.x == 0) {
                printHash("hashb3-1 is : ", hash_.hash, 32);
            }
            */
            
           fishhash_context ctx {
                LIGHT_CACHE_NUM_ITEMS,
                cache,
                FULL_DATASET_NUM_ITEMS,
                dataset
            };

            memset(input, 0, 80);
            memcpy(input, hash_.hash, 32);
            hashFish(&ctx, hash_.hash, input);

            /*
            if (threadIdx.x == 0 && blockIdx.x == 0) {
                printHash("hashFish is : ", hash_.hash, 32);
            }
            */

            memset(input, 0, 80);
            memcpy(input, hash_.hash, 32);
            hashB3(hash_.hash, input, 32);

            /*
            if (threadIdx.x == 0 && blockIdx.x == 0) {
                printHash("hashb3-2 is : ", hash_.hash, 32);
            }
            */
            
            
            if (LT_U256(hash_, target)){
                atomicCAS((unsigned long long int*) final_nonce, 0, (unsigned long long int) nonce);
            }
        }
    }

}