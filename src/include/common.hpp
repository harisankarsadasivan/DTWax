#ifndef COMMON_HPP
#define COMON_HPP

//...............global variables..........................//
#ifdef FP16

#include <cuda_fp16.h>
typedef __half2 value_ht;
#define HALF2FLOAT(a) __half2float(a)
#define FLOAT2HALF(a, b) __floats2half2_rn(a, b)
#define FIND_MIN(a, b) __hmin2(a, b)
#define FMA(a, b, c) __hfma2(a, b, c)
#define ADD(a, b) __hadd2(a, b)
#define DIV(a, b) __h2div(a, b)
#define SQRT(a) h2sqrt(a)
#define FLOAT2HALF2(a) FLOAT2HALF(a, a)

#else
typedef float value_ht;
#define FLOAT2HALF(a) a
#define HALF2FLOAT(a) a
#define FIND_MIN(a, b) min(a, b)
#define FIND_MAX(a, b) max(a, b)
#define FMA(a, b, c) __fmaf_ieee_rn(a, b, c)
#define ADD(a, b) (a + b)
#define DIV(a, b) (a & (b - 1)) // make sure b is power of 2
#define SQRT(a) sqrtf(a)        // a is to be float
#define FLOAT2HALF2(a) FLOAT2HALF(a)
#endif

#define KMER_LEN 6
#define WARP_SIZE 32
#define SEGMENT_SIZE 1
#define LOG_WARP_SIZE 5
#define QUERY_LEN (32) //>=WARP_SIZE for the coalesced shared mem
// #define REF_LEN 48502

#ifndef FP16
#define REF_LEN                                                                \
  (64) // indicates total length of forward + backward squiggle
       // genome ; should be a multiple of SEGMENT_SIZE*WARP_SIZE*2
#else
#define REF_LEN (64) // length of fwd strand in case of FP16
#endif

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
#define WARP_SIZE_MINUS_ONE (WARP_SIZE - 1)
#define RESULT_REG (SEGMENT_SIZE - 1)
#define NUM_WAVES_BY_WARP_SIZE ((NUM_WAVES / WARP_SIZE) * WARP_SIZE)

#endif