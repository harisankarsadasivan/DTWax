#ifndef COMMON_HPP
#define COMON_HPP

//...............global variables..........................//
#ifdef FP16

#include <cuda_fp16.h>
typedef __half2 value_ht;
#define FP_PIPES 2
#define HALF2FLOAT(a) __half2float(a)
#define FLOAT2HALF(a) __float2half2_rn(a)
#define FIND_MIN(a, b) __hmin2(a, b)
#define FMA(a, b, c) __hfma2(a, b, c)
#define ADD(a, b) __hadd2(a, b)
#define DIV(a, b) __h2div(a, b)
#define SQRT(a) h2sqrt(a)

#else

#define FP_PIPES 1
#define FLOAT2HALF(a) a
#define HALF2FLOAT(a) a
typedef float value_ht;
#define FIND_MIN(a, b) min(a, b)
#define FIND_MAX(a, b) max(a, b)
#define FMA(a, b, c) (a * b + c)
#define ADD(a, b) (a + b)
#define DIV(a, b) (a & (b - 1)) // make sure b is power of 2
#define SQRT(a) sqrtf(a)        // a is to be float
#endif

#define KMER_LEN 6
#define WARP_SIZE 32
#define SEGMENT_SIZE 1
#define LOG_WARP_SIZE 5
#define QUERY_LEN (32)
// #define REF_LEN 48502
#define REF_LEN (32) // change this to original length later; Keep this "even"!
#define BLOCK_NUM (1)
#define STREAM_NUM 1

#define ADAPTER_LEN 1000
#define ONT_FILE_FORMAT "fast5"

//-----------------derived variables--------------------------//

#define GROUP_SIZE WARP_SIZE
#define REF_TILE_SIZE (SEGMENT_SIZE * WARP_SIZE)
#define REF_BATCH (REF_LEN / REF_TILE_SIZE)

/* calculate when to stop, and which thread has final result */
#define NUM_WAVES (QUERY_LEN + (REF_TILE_SIZE - 1) / (SEGMENT_SIZE))
#define RESULT_THREAD_ID (WARP_SIZE - 1)
#define RESULT_REG (SEGMENT_SIZE - 1)

#endif