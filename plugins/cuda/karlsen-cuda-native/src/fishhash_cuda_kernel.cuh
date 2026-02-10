#include "fishhash_cuda_kernel.h"
#include "keccak.cuh"

#define FNV_PRIME 0x01000193
#define FULL_DATASET_ITEM_PARENTS 512
#define NUM_DATASET_ACCESSES 32  
#define LIGHT_CACHE_ROUNDS 3
#define LIGHT_CACHE_NUM_ITEMS 1179641
#define FULL_DATASET_NUM_ITEMS 37748717

#define DEV_INLINE __device__ __forceinline__

#define copy(dst, src, count)        \
    for (int i = 0; i != count; ++i) \
    {                                \
        (dst)[i] = (src)[i];         \
    }

#define fnv(x, y)        ((x) * FNV_PRIME ^ (y))
#define fnv_reduce(v)    fnv(fnv(fnv(v.x, v.y), v.z), v.w)

DEV_INLINE hash512 fnv1(const hash512& u, const hash512& v) noexcept {
    hash512 r;
    for (size_t i = 0; i < sizeof(r) / sizeof(r.word32s[0]); ++i)
        r.word32s[i] = fnv(u.word32s[i], v.word32s[i]);
    return r;
}

typedef struct item_state
	{
	    const hash512* const cache;
	    const int64_t num_cache_items;
	    const uint32_t seed;

	    hash512 mix;

	    DEV_INLINE item_state(const fishhash_context& ctx, int64_t index) noexcept
	      : cache{ctx.light_cache},
		    num_cache_items{ctx.light_cache_num_items},
		    seed{static_cast<uint32_t>(index)} {
		    mix = cache[index % num_cache_items];
		    mix.word32s[0] ^= seed;
            SHA3_512(mix.uint2s);
	    }

	    DEV_INLINE void update(uint32_t round) noexcept {
		    static constexpr size_t num_words = sizeof(mix) / sizeof(uint32_t);
		    const uint32_t t = fnv(seed ^ round, mix.word32s[round % num_words]);
		    const int64_t parent_index = t % num_cache_items;
		    mix = fnv1(mix, cache[parent_index]);
	    }

	    DEV_INLINE hash512 final() noexcept { 
            SHA3_512(mix.uint2s);
	    	return mix; 
	    }
	} item_state;



DEV_INLINE hash1024 calculate_dataset_item_1024(const fishhash_context& ctx, uint32_t index) noexcept {
    item_state item0{ctx, int64_t(index) * 2};
    item_state item1{ctx, int64_t(index) * 2 + 1};

	for (uint32_t j = 0; j < FULL_DATASET_ITEM_PARENTS; ++j) {
		item0.update(j);
		item1.update(j);
	}

    hash512 it0 = item0.final();
    hash512 it1 = item1.final();

    return hash1024{{it0, it1}};
}

DEV_INLINE hash1024 lookup(const fishhash_context& ctx, uint32_t index) {
    if (ctx.full_dataset != NULL) {
        hash1024 * item = &ctx.full_dataset[index];
        
        // Ability to handle lazy lookup
        if (item->word64s[0] == 0) {
            *item = calculate_dataset_item_1024(ctx, index);
        }
        
        return *item;
    } else {
        return calculate_dataset_item_1024(ctx, index);
    }
}

DEV_INLINE hash256 fishhash_kernel(const fishhash_context& ctx, const hash512& seed) noexcept {
		const uint32_t index_limit = static_cast<uint32_t>(ctx.full_dataset_num_items);
		hash1024 mix{seed, seed};
		
		#pragma unroll
		for (uint32_t i = 0; i < NUM_DATASET_ACCESSES; ++i) {
			uint32_t mixGroup[8]; 

			#pragma unroll
			for (uint32_t c = 0; c < 8; c++) {
				mixGroup[c] = (mix.word32s[4*c + 0] ^ mix.word32s[4*c + 1] ^ mix.word32s[4*c + 2] ^ mix.word32s[4*c + 3]);
			}

			const uint32_t p0 = (mixGroup[0] ^ mixGroup[3] ^ mixGroup[6]) % index_limit;
			const uint32_t p1 = (mixGroup[1] ^ mixGroup[4] ^ mixGroup[7]) % index_limit;
			const uint32_t p2 = (mixGroup[2] ^ mixGroup[5] ^           i) % index_limit;

			hash1024 fetch0 = lookup(ctx, p0);
			hash1024 fetch1 = lookup(ctx, p1);
			hash1024 fetch2 = lookup(ctx, p2);

			// Modify fetch1 and fetch2
			#pragma unroll
			for (size_t j = 0; j < 32; ++j) {
				fetch1.word32s[j] = fnv(mix.word32s[j], fetch1.word32s[j]);
				fetch2.word32s[j] = mix.word32s[j] ^ fetch2.word32s[j];
			}

	     	// Final computation of new mix
			#pragma unroll
			for (size_t j = 0; j < 16; ++j) {
				mix.word64s[j] = fetch0.word64s[j] * fetch1.word64s[j] + fetch2.word64s[j];
			}
		}

		// Collapse the result into 32 bytes
		hash256 mix_hash;
		static constexpr size_t num_words = sizeof(mix) / sizeof(uint32_t);
		
		#pragma unroll
		for (size_t i = 0; i < num_words; i += 4) {
			uint4 vec = make_uint4(mix.word32s[i], mix.word32s[i + 1], mix.word32s[i + 2], mix.word32s[i + 3]);
			mix_hash.word32s[i / 4] = fnv_reduce(vec);
		}

		return mix_hash;
	}

DEV_INLINE void printHash(char* msg, const uint8_t* hash, int size) {
		printf(msg);
		for(int i = 0; i < size; i++) {
			printf("%02x", hash[i]);
		}
		printf("\n");
	}

DEV_INLINE void hashFish(
            const fishhash_context * ctx,
            uint8_t* out,
            const uint8_t* in) {
		hash512 seed; 
		memset(seed.bytes, 0, 64);
		memcpy(seed.bytes, in, 32);

		const hash256 mix_hash = fishhash_kernel(*ctx, seed);
		memcpy(out, mix_hash.bytes, 32);
	}



DEV_INLINE hash512 bitwise_xor(const hash512& x, const hash512& y) noexcept {
		hash512 z;
		for (size_t i = 0; i < sizeof(z) / sizeof(z.word64s[0]); ++i)
			z.word64s[i] = x.word64s[i] ^ y.word64s[i];
		return z;
	}

