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
__device__ bool b58enc(char* b58, size_t* b58sz, const uint8_t* data, size_t binsz);

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

__device__ bool check_pattern(const char* address, const pattern_t* pattern) {
    switch (pattern->type) {
        case PATTERN_TYPE_STARTS_WITH:
            return check_starts_with(address, pattern->pattern);
            
        case PATTERN_TYPE_ENDS_WITH:
            return check_ends_with(address, pattern->pattern);
            
        case PATTERN_TYPE_STARTS_AND_ENDS_WITH:
            return check_starts_and_ends_with(address, pattern->pattern, pattern->end_pattern);
            
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

void vanity_run(config &vanity) {
    int gpuCount = 0;
    cudaGetDeviceCount(&gpuCount);

    unsigned long long executions_total = 0;
    unsigned long long keys_found_total = 0;

    const int maxDevices = 100;
    // Arrays to hold per-GPU device pointer for counters and results.
    int* dev_executions_this_gpu[maxDevices] = {0};
    int* dev_keys_found[maxDevices] = {0};
    int* dev_resultCount[maxDevices] = {0};
    KeyRecord* dev_results[maxDevices] = {0};

    // For each iteration:
    for (int iter = 0; iter < MAX_ITERATIONS; ++iter) {
        auto start = std::chrono::high_resolution_clock::now();
        unsigned long long executions_this_iteration = 0;
        unsigned long long keys_found_this_iteration = 0;

        // For each GPU in the iteration:
        for (int g = 0; g < gpuCount; ++g) {
            cudaSetDevice(g);

            // Compute grid and block sizes for the kernel.
            int blockSize = 0, minGridSize = 0, maxActiveBlocksPerSM = 0;
            cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, vanity_scan, 0, 0);
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocksPerSM, vanity_scan, blockSize, 0);
            cudaDeviceProp devProp;
            cudaGetDeviceProperties(&devProp, g);
            int totalBlocks = maxActiveBlocksPerSM * devProp.multiProcessorCount;

            // Reseed the curand state on this GPU.
            unsigned long long int new_seed = makeSeed();
            unsigned long long int* dev_new_seed = nullptr;
            cudaMalloc((void**)&dev_new_seed, sizeof(unsigned long long int));
            cudaMemcpy(dev_new_seed, &new_seed, sizeof(unsigned long long int), cudaMemcpyHostToDevice);
            vanity_init<<<totalBlocks, blockSize>>>(dev_new_seed, vanity.states[g]);
            cudaFree(dev_new_seed);

            // Allocate a device int for the GPU id.
            int* dev_g = nullptr;
            cudaMalloc((void**)&dev_g, sizeof(int));
            cudaMemcpy(dev_g, &g, sizeof(int), cudaMemcpyHostToDevice);

            // Allocate and zero-initialize the per-GPU counters.
            cudaMalloc((void**)&dev_keys_found[g], sizeof(int));
            cudaMalloc((void**)&dev_executions_this_gpu[g], sizeof(int));
            cudaMalloc((void**)&dev_resultCount[g], sizeof(int));
            cudaMemset(dev_keys_found[g], 0, sizeof(int));
            cudaMemset(dev_executions_this_gpu[g], 0, sizeof(int));
            cudaMemset(dev_resultCount[g], 0, sizeof(int));

            // Allocate the results buffer (an array of KeyRecord).
            const int MAX_RESULTS = 10000;
            cudaMalloc((void**)&dev_results[g], sizeof(KeyRecord) * MAX_RESULTS);

            // Launch the kernel on GPU g.
            vanity_scan<<<totalBlocks, blockSize>>>(vanity.states[g],
                                                      dev_g,
                                                      dev_executions_this_gpu[g],
                                                      dev_keys_found[g],
                                                      dev_resultCount[g],
                                                      dev_results[g]);
            cudaFree(dev_g);
        }


        // Wait for all kernels to complete.
        cudaDeviceSynchronize();
        auto finish = std::chrono::high_resolution_clock::now();

        // Copy back the per-GPU counters.
        for (int g = 0; g < gpuCount; ++g) {
            int temp = 0;
            cudaMemcpy(&temp, dev_keys_found[g], sizeof(int), cudaMemcpyDeviceToHost);
            keys_found_this_iteration += temp;

            int gpu_exec = 0;
            cudaMemcpy(&gpu_exec, dev_executions_this_gpu[g], sizeof(int), cudaMemcpyDeviceToHost);
            executions_this_iteration += (unsigned long long)gpu_exec * ATTEMPTS_PER_EXECUTION;

            // Free per-GPU allocations.
            cudaFree(dev_keys_found[g]);
            cudaFree(dev_executions_this_gpu[g]);
            cudaFree(dev_resultCount[g]);
            cudaFree(dev_results[g]);
        }

        executions_total += executions_this_iteration;
        keys_found_total += keys_found_this_iteration;

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
    // Calculate unique thread id.
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    atomicAdd(exec_count, 1);
    
    // Get the thread's random state.
    curandState localState = state[id];

    // ---------------------------------------------------------------
    // Generate a high-entropy base seed unique to this thread.
    // We mix in the thread id so different threads cover different regions.
    // ---------------------------------------------------------------
    unsigned char seed[32];
    for (int i = 0; i < 32; i++) {
        // Mix a random byte from curand with some bits from the thread id.
        seed[i] = (uint8_t)(curand(&localState) & 0xFF) ^ ((id >> (i % 8)) & 0xFF);
    }
    
    // Prepare a constant “one” that will be added to the seed each attempt.
    // (This is our deterministic step: simply add 1 as a big-integer.)
    unsigned char one[32] = {0};
    one[31] = 1;  // least significant byte = 1

    ge_p3 A;
    unsigned char publick[32];
    unsigned char privatek[64];
    char key[KEY_STRING_SIZE];

    // Determine number of patterns (assume patterns and MAX_PATTERNS are defined)
    int numPatterns = sizeof(patterns) / sizeof(pattern_t);
    int prefix_letter_counts[MAX_PATTERNS];
    for (int p = 0; p < numPatterns; p++) {
        int count = 0;
        while (patterns[p].pattern[count] != '\0') count++;
        prefix_letter_counts[p] = count;
    }
    
    // ------------------------------------------------------------------
    // Now, in each iteration we derive keys based on the current seed.
    // Then we increment the seed deterministically to cover the space.
    // ------------------------------------------------------------------
    for (int attempt = 0; attempt < ATTEMPTS_PER_EXECUTION; ++attempt) {
        // Derive the 64-byte hash from the 32-byte seed.
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

        // Convert the public key to a Base58-encoded address.
        size_t keysize = KEY_STRING_SIZE;
        b58enc(key, &keysize, publick, 32);

        // Check each pattern.
        for (int p = 0; p < numPatterns; ++p) {
            bool match = true;
            for (int j = 0; j < prefix_letter_counts[p]; ++j) {
                if ((patterns[p].pattern[j] != '?') && (patterns[p].pattern[j] != key[j])) {
                    match = false;
                    break;
                }
            }
            if (match) {
                // If we have a match, update counters and record the result.
                atomicAdd(keys_found, 1);
                int index = atomicAdd(resultCount, 1);
                for (int j = 0; j < KEY_STRING_SIZE; j++) {
                    results[index].key[j] = key[j];
                }
                for (int j = 0; j < SEED_SIZE; j++) {
                    results[index].seed[j] = seed[j];
                }
                // Build the full 64-byte private key:
                unsigned char fullPrivate[64];
                for (int n = 0; n < SEED_SIZE; n++) {
                    fullPrivate[n] = seed[n];
                }
                for (int n = 0; n < 32; n++) {
                    fullPrivate[n + SEED_SIZE] = publick[n];
                }
                // Print the result in JSON format.
                printf("{\"address\":\"%s\",\"private_key\":[", key);
                for (int n = 0; n < 64; n++) {
                    printf("%d", fullPrivate[n]);
                    if (n < 63) { printf(","); }
                }
                printf("]}\n");
                break; // Stop checking further patterns for this attempt.
            }
        }
        
        // Increment the seed as a big-integer (seed = seed + 1).
        int carry = 0;
        for (int i = 31; i >= 0; i--) {
            int sum = (int)seed[i] + (int)one[i] + carry;
            seed[i] = (unsigned char)(sum & 0xFF);
            carry = sum >> 8;
        }
    }
    
    // Write back the updated curand state.
    state[id] = localState;
}

__device__ bool b58enc(char* b58, size_t* b58sz, const uint8_t* data, size_t binsz) {
    static const char b58digits_ordered[] = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

    int carry;
    size_t i, j, high, zcount = 0;
    size_t size = (binsz * 138 / 100) + 1;
    uint8_t buf[256] = {0}; // Initialize to zero directly.

    while (zcount < binsz && !data[zcount])
        ++zcount;
    
    size -= zcount; // Adjust size for leading zeros.
    
    for (i = zcount, high = size - 1; i < binsz; ++i, high = j) {
        carry = data[i];
        for (j = size - 1; j > high || carry; --j) {
            carry += 256 * buf[j];
            buf[j] = carry % 58;
            carry /= 58;
        }
    }
    
    for (j = 0; j < size && !buf[j]; ++j);
    
    if (*b58sz <= zcount + size - j) {
        *b58sz = zcount + size - j + 1;
        return false;
    }
    
    if (zcount) memset(b58, '1', zcount);
    for (i = zcount; j < size; ++i, ++j) {
        b58[i] = b58digits_ordered[buf[j]];
    }
    b58[i] = '\0';
    *b58sz = i + 1;
    
    return true;
}
