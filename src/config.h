#ifndef VANITY_CONFIG
#define VANITY_CONFIG

static int const MAX_ITERATIONS = 100000;
static int const STOP_AFTER_KEYS_FOUND = 100;

// how many times a gpu thread generates a public key in one go
__device__ const int ATTEMPTS_PER_EXECUTION = 100000;

__device__ const int MAX_PATTERNS = 10;

// Pattern types
#define PATTERN_TYPE_STARTS_WITH 0
#define PATTERN_TYPE_ENDS_WITH 1  
#define PATTERN_TYPE_STARTS_AND_ENDS_WITH 2

// Pattern struct to hold type and pattern
typedef struct {
    int type;
    char pattern[32];  // Max pattern length
    char end_pattern[32];  // For starts_and_ends_with
} pattern_t;

#define KEY_STRING_SIZE 256
#define SEED_SIZE 32

typedef struct {
    char key[KEY_STRING_SIZE];      // The Base58-encoded key
    unsigned char seed[SEED_SIZE];    // The seed that produced it
} KeyRecord;


// Array of patterns to match
__device__ static pattern_t patterns[] = {
    // Example patterns - modify as needed
    {PATTERN_TYPE_STARTS_WITH, "AAAA"},
    //{PATTERN_TYPE_ENDS_WITH, "ZZZZ"},
    //{PATTERN_TYPE_STARTS_AND_ENDS_WITH, "bob", "bob"}
};

#endif
