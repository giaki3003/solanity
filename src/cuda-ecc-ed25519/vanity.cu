#include <vector>
#include <random>
#include <chrono>
#include <iostream>
#include <ctime>
#include <assert.h>
#include <inttypes.h>
#include <pthread.h>
#include <stdio.h>

#include "curand_kernel.h"
#include "ed25519.h"
#include "fixedint.h"
#include "gpu_common.h"
#include "gpu_ctx.h"

#include "keypair.cu"
#include "sc.cu"
#include "fe.cu"
#include "ge.cu"
#include "sha512.cu"
#include "../config.h"

/* -- Types ----------------------------------------------------------------- */

typedef struct {
    curandState* states[8];
} config;

/* -- Prototypes ------------------------------------------------------------ */

void            vanity_setup(config& vanity);
void            vanity_run(config& vanity);
void __global__ vanity_init(unsigned long long int* seed, curandState* state);
void __global__ vanity_scan(curandState* state, int* gpu, int* exec_count, int* keys_found, int* resultCount, KeyRecord* results);
bool __device__ b58enc(char* b58, size_t* b58sz, uint8_t* data, size_t binsz);

/* -- Pattern Matching Functions -------------------------------------------- */

__device__ bool check_starts_with(const char* address, const char* pattern) {
    while (*pattern) {
        if (*pattern != '?' && *pattern != *address) {
            return false;
        }
        pattern++;
        address++;
    }
    return true;
}

__device__ bool check_ends_with(const char* address, const char* pattern) {
    size_t addr_len = 0;
    size_t pattern_len = 0;
    
    // Get lengths
    while (address[addr_len]) addr_len++;
    while (pattern[pattern_len]) pattern_len++;
    
    if (pattern_len > addr_len) return false;
    
    const char* addr_end = address + addr_len - pattern_len;
    while (*pattern) {
        if (*pattern != '?' && *pattern != *addr_end) {
            return false;
        }
        pattern++;
        addr_end++;
    }
    return true;
}

__device__ bool check_starts_and_ends_with(const char* address, const char* start_pattern, const char* end_pattern) {
    return check_starts_with(address, start_pattern) && check_ends_with(address, end_pattern);
}

__device__ bool check_contains_mnemonic(const char* address) {
    for (int i = 0; i < sizeof(mnemonic_words) / sizeof(mnemonic_words[0]); i++) {
        const char* word = mnemonic_words[i];
        const char* found = address;
        while (*found) {
            bool match = true;
            const char* w = word;
            const char* f = found;
            while (*w) {
                if (*w != *f) {
                    match = false;
                    break;
                }
                w++;
                f++;
            }
            if (match) return true;
            found++;
        }
    }
    return false;
}

__device__ bool check_pattern(const char* address, const pattern_t* pattern) {
    switch (pattern->type) {
        case PATTERN_TYPE_STARTS_WITH:
            return check_starts_with(address, pattern->pattern);
            
        case PATTERN_TYPE_ENDS_WITH:
            return check_ends_with(address, pattern->pattern);
            
        case PATTERN_TYPE_STARTS_AND_ENDS_WITH:
            return check_starts_and_ends_with(address, pattern->pattern, pattern->end_pattern);
            
        case PATTERN_TYPE_MNEMONIC:
            return check_contains_mnemonic(address);
            
        default:
            return false;
    }
}

/* -- Entry Point ----------------------------------------------------------- */

int main(int argc, char const* argv[]) {
    ed25519_set_verbose(true);

    config vanity;
    vanity_setup(vanity);
    vanity_run(vanity);
}

// SMITH
std::string getTimeStr() {
    std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::string s(30, '\0');
    std::strftime(&s[0], s.size(), "%Y-%m-%d %H:%M:%S", std::localtime(&now));
    return s;
}

// SMITH - safe? who knows
unsigned long long int makeSeed() {
    unsigned long long int seed = 0;
    char *pseed = (char *)&seed;

    std::random_device rd;

    for(unsigned int b=0; b<sizeof(seed); b++) {
      auto r = rd();
      char *entropy = (char *)&r;
      pseed[b] = entropy[0];
    }

    return seed;
}

/* -- Vanity Setup Function ------------------------------------------------- */
void vanity_setup(config &vanity) {
    printf("GPU: Initializing Memory\n");
    int gpuCount = 0;
    cudaGetDeviceCount(&gpuCount);

    for (int i = 0; i < gpuCount; ++i) {
        cudaSetDevice(i);
        cudaDeviceProp device;
        cudaGetDeviceProperties(&device, i);

        int blockSize = 0, minGridSize = 0, maxActiveBlocksPerSM = 0;
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, vanity_scan, 0, 0);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocksPerSM, vanity_scan, blockSize, 0);

        int totalBlocks = maxActiveBlocksPerSM * device.multiProcessorCount;
        int totalThreads = totalBlocks * blockSize;
        printf("GPU %d: totalThreads = %d\n", i, totalThreads);

        unsigned long long int rseed = makeSeed();
        printf("Initialising from entropy: %llu\n", rseed);

        unsigned long long int* dev_rseed;
        cudaMalloc((void**)&dev_rseed, sizeof(unsigned long long int));
        cudaMemcpy(dev_rseed, &rseed, sizeof(unsigned long long int), cudaMemcpyHostToDevice);

        // Allocate enough curandState for all threads.
        cudaMalloc((void **)&(vanity.states[i]), totalThreads * sizeof(curandState));
        vanity_init<<<maxActiveBlocksPerSM, blockSize>>>(dev_rseed, vanity.states[i]);
        cudaFree(dev_rseed);
    }
    printf("END: Initializing Memory\n");
}

/* -- Vanity Run Function --------------------------------------------------- */
void vanity_run(config &vanity) {
    int gpuCount = 0;
    cudaGetDeviceCount(&gpuCount);

    // Overall counters
    unsigned long long executions_total = 0;
    unsigned long long keys_found_total = 0;

    // We assume that no more than maxDevices GPUs will be used.
    const int maxDevices = 100;
    // Arrays to hold pointers to per-GPU device counters and result buffers.
    int* dev_executions_this_gpu[maxDevices] = {0};
    int* dev_keys_found[maxDevices] = {0};
    int* dev_resultCount[maxDevices] = {0};
    KeyRecord* dev_results[maxDevices] = {0};

    // Loop over iterations
    for (int iter = 0; iter < MAX_ITERATIONS; ++iter) {
        auto start = std::chrono::high_resolution_clock::now();

        unsigned long long executions_this_iteration = 0;
        unsigned long long keys_found_this_iteration = 0;

        // Launch kernels on every GPU.
        for (int g = 0; g < gpuCount; ++g) {
            cudaSetDevice(g);

            int blockSize = 0, minGridSize = 0, maxActiveBlocksPerSM = 0;
            // Determine occupancy parameters for the vanity_scan kernel.
            cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, vanity_scan, 0, 0);
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocksPerSM, vanity_scan, blockSize, 0);
            cudaDeviceProp devProp;
            cudaGetDeviceProperties(&devProp, g);
            int totalBlocks = maxActiveBlocksPerSM * devProp.multiProcessorCount;

            // Allocate a device int holding the GPU id.
            int* dev_g = nullptr;
            cudaMalloc((void**)&dev_g, sizeof(int));
            cudaMemcpy(dev_g, &g, sizeof(int), cudaMemcpyHostToDevice);

            // Allocate and initialize the counters.
            cudaMalloc((void**)&dev_keys_found[g], sizeof(int));
            cudaMalloc((void**)&dev_executions_this_gpu[g], sizeof(int));
            cudaMalloc((void**)&dev_resultCount[g], sizeof(int));
            cudaMemset(dev_keys_found[g], 0, sizeof(int));
            cudaMemset(dev_executions_this_gpu[g], 0, sizeof(int));
            cudaMemset(dev_resultCount[g], 0, sizeof(int));

            // Allocate the result buffer for matching keys.
            const int MAX_RESULTS = 10000;
            cudaMalloc((void**)&dev_results[g], sizeof(KeyRecord) * MAX_RESULTS);

            // Launch the kernel on GPU g.
            // The kernel has 6 parameters:
            // 1. the curandState array,
            // 2. pointer to GPU id,
            // 3. pointer to execution count,
            // 4. pointer to keys-found count,
            // 5. pointer to the resultCount counter,
            // 6. pointer to the KeyRecord results array.
            vanity_scan<<<totalBlocks, blockSize>>>(vanity.states[g],
                                                      dev_g,
                                                      dev_executions_this_gpu[g],
                                                      dev_keys_found[g],
                                                      dev_resultCount[g],
                                                      dev_results[g]);
            cudaFree(dev_g);
        }

        cudaDeviceSynchronize();
        auto finish = std::chrono::high_resolution_clock::now();

        // Collect the per-GPU counters.
        for (int g = 0; g < gpuCount; ++g) {
            int gpu_keys = 0;
            cudaMemcpy(&gpu_keys, dev_keys_found[g], sizeof(int), cudaMemcpyDeviceToHost);
            keys_found_this_iteration += gpu_keys;
            keys_found_total += gpu_keys;

            int gpu_exec = 0;
            cudaMemcpy(&gpu_exec, dev_executions_this_gpu[g], sizeof(int), cudaMemcpyDeviceToHost);
            executions_this_iteration += (unsigned long long)gpu_exec * ATTEMPTS_PER_EXECUTION;
            executions_total += (unsigned long long)gpu_exec * ATTEMPTS_PER_EXECUTION;

            cudaFree(dev_keys_found[g]);
            cudaFree(dev_executions_this_gpu[g]);
            cudaFree(dev_resultCount[g]);
            cudaFree(dev_results[g]);
        }

        std::chrono::duration<double> elapsed = finish - start;
        printf("%s Iteration %d Attempts: %llu in %f sec at %fcps - Total Attempts %llu - keys found %llu\n",
           getTimeStr().c_str(),
           iter + 1,
           executions_this_iteration,
           elapsed.count(),
           executions_this_iteration / elapsed.count(),
           executions_total,
           keys_found_total);

        if (keys_found_total >= STOP_AFTER_KEYS_FOUND) {
            printf("Enough keys found, Done!\n");
            exit(0);
        }
    }

    printf("Iterations complete, Done!\n");
}

/* -- CUDA Vanity Functions ------------------------------------------------- */

__global__ void vanity_init(unsigned long long int* rseed, curandState* state) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(*rseed + id, id, 0, &state[id]);
}

__global__ void vanity_scan(curandState* state,
                              int* gpu,             // GPU id pointer
                              int* exec_count,      // Execution counter
                              int* keys_found,      // Keys found counter
                              int* resultCount,     // Atomic counter for results
                              KeyRecord* results)   // Global array for matching keys
{
    // Compute a unique thread id.
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    atomicAdd(exec_count, 1);

    // Compute the number of patterns and their lengths.
    int numPatterns = sizeof(patterns) / sizeof(pattern_t);
    int prefix_letter_counts[MAX_PATTERNS];
    for (int p = 0; p < numPatterns; p++) {
        int count = 0;
        while (patterns[p].pattern[count] != '\0') {
            count++;
        }
        prefix_letter_counts[p] = count;
    }

    // Local ED25519 state and random state.
    ge_p3 A;
    curandState localState = state[id];
    unsigned char seed[32] = {0};
    unsigned char publick[32] = {0};
    unsigned char privatek[64] = {0};
    char key[KEY_STRING_SIZE] = {0};

    // Initialize the seed from curand.
    for (int i = 0; i < 32; ++i) {
        float rnd = curand_uniform(&localState);
        seed[i] = (uint8_t)(rnd * 255);
    }

    // Main loop: try ATTEMPTS_PER_EXECUTION times.
    for (int attempt = 0; attempt < ATTEMPTS_PER_EXECUTION; ++attempt) {
        // --- Derive the key via SHA512 (inlined) ---
        sha512_context md;
        sha512_init(&md);
        sha512_update(&md, seed, 32);
        sha512_final(&md, privatek);

        // ED25519 key clamping.
        privatek[0]  &= 248;
        privatek[31] &= 63;
        privatek[31] |= 64;

        // Compute the public key.
        ge_scalarmult_base(&A, privatek);
        ge_p3_tobytes(publick, &A);

        // Convert public key to a Base58 address.
        size_t keysize = KEY_STRING_SIZE;
        b58enc(key, &keysize, publick, 32);

        // --- Check all patterns ---
        for (int p = 0; p < numPatterns; ++p) {
            bool match = true;
            for (int j = 0; j < prefix_letter_counts[p]; ++j) {
                if ((patterns[p].pattern[j] != '?') && (patterns[p].pattern[j] != key[j])) {
                    match = false;
                    break;
                }
            }
            if (match) {
                // A match was found.
                atomicAdd(keys_found, 1);

                int index = atomicAdd(resultCount, 1);
                for (int j = 0; j < KEY_STRING_SIZE; j++) {
                    results[index].key[j] = key[j];
                }
                for (int j = 0; j < SEED_SIZE; j++) {
                    results[index].seed[j] = seed[j];
                }

                // Build the full 64-byte private key (seed || public key)
                unsigned char fullPrivate[64];
                for (int n = 0; n < SEED_SIZE; n++) {
                    fullPrivate[n] = seed[n];
                }
                for (int n = 0; n < 32; n++) {
                    fullPrivate[n + SEED_SIZE] = publick[n];
                }
                // Print the JSON object with no extraneous whitespace.
                printf("{\"address\":\"%s\",\"private_key\":[", key);
                for (int n = 0; n < 64; n++) {
                    printf("%d", fullPrivate[n]);
                    if (n < 63)
                        printf(",");
                }
                printf("]}\n");
                break; // Stop checking patterns for this attempt.
            }
        }

        // Increment seed using simple counter logic.
        for (int i = 0; i < 32; ++i) {
            if (seed[i] == 255)
                seed[i] = 0;
            else {
                seed[i] += 1;
                break;
            }
        }
    }

    // Save the updated curand state.
    state[id] = localState;
}

bool __device__ b58enc(char* b58, size_t* b58sz, uint8_t* data, size_t binsz) {
    const char b58digits_ordered[] = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

    const uint8_t* bin = data;
    int carry;
    size_t i, j, high, zcount = 0;
    size_t size;
    
    while (zcount < binsz && !bin[zcount])
        ++zcount;
    
    size = (binsz - zcount) * 138 / 100 + 1;
    uint8_t buf[256];
    memset(buf, 0, size);
    
    for (i = zcount, high = size - 1; i < binsz; ++i, high = j) {
        for (carry = bin[i], j = size - 1; (j > high) || carry; --j) {
            carry += 256 * buf[j];
            buf[j] = carry % 58;
            carry /= 58;
            if (!j) {
                break;
            }
        }
    }
    
    for (j = 0; j < size && !buf[j]; ++j);
    
    if (*b58sz <= zcount + size - j) {
        *b58sz = zcount + size - j + 1;
        return false;
    }
    
    if (zcount) memset(b58, '1', zcount);
    for (i = zcount; j < size; ++i, ++j) b58[i] = b58digits_ordered[buf[j]];
    b58[i] = '\0';
    *b58sz = i + 1;
    
    return true;
}