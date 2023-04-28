/*
// Not a contribution
// Changes made by NVIDIA CORPORATION & AFFILIATES enabling <XYZ> or otherwise
documented as
// NVIDIA-proprietary are not a contribution and subject to the following terms
and conditions:
 * SPDX-FileCopyrightText: Copyright (c) <year> NVIDIA CORPORATION & AFFILIATES.
All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.

 # SPDX-FileCopyrightText: Copyright (c) <year> NVIDIA CORPORATION & AFFILIATES.
All rights reserved. # SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 #
 # NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 # property and proprietary rights in and to this material, related
 # documentation and any modifications thereto. Any use, reproduction,
 # disclosure or distribution of this material and related documentation
 # without an express license agreement from NVIDIA CORPORATION or
 # its affiliates is strictly prohibited.
 */
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
#define SUB(a, b) __hsub2(a, b)
#define SQRT(a) h2sqrt(a)
#define FLOAT2HALF2(a) FLOAT2HALF(a, a)
#define GT(a, b) __hgt(a, b)

#else
typedef float value_ht;
#define FLOAT2HALF(a) a
#define HALF2FLOAT(a) a
#define FIND_MIN(a, b) min(a, b)
#define FIND_MAX(a, b) max(a, b)
#define FMA(a, b, c) __fmaf_ieee_rn(a, b, c)
#define ADD(a, b) (a + b)
#define SUB(a, b) (a - b) // make sure b is power of 2
#define SQRT(a) sqrtf(a)  // a is to be float
#define FLOAT2HALF2(a) FLOAT2HALF(a)
#define GT(a, b) (a > b)
#endif

#define KMER_LEN 6
#define WARP_SIZE 32
// 32 seems to produce results more closer to python with less divergence
#define LOG_WARP_SIZE 5
#define SEGMENT_SIZE 26 // 40
#define QUERY_LEN (512 * 4)
#define PREFIX_LEN (512 * 2)
//>=WARP_SIZE for the coalesced shared mem; has to be a multiple of 32; >=64 if
// using PINGPONG buffer

#ifndef FP16
#define REF_LEN                                                                \
  (47 * 1024 * 2) // indicates total length of forward + backward squiggle
// genome ; should be a multiple of SEGMENT_SIZE*WARP_SIZE
#else
#define REF_LEN (47 * 1024) // length of fwd strand in case of FP16
#endif

#define BLOCK_NUM (10)
#define STREAM_NUM 1
// #define SMEM_BUFFER_SIZE 32 // has to be a multiple of 2*WARP_SIZE

#define ADAPTER_LEN 1952 // change for Read Until
#define ONT_FILE_FORMAT "fast5"

//-----------------derived variables--------------------------//

#define REF_TILE_SIZE (SEGMENT_SIZE * WARP_SIZE)
#define REF_BATCH (REF_LEN / REF_TILE_SIZE)
#define QUERY_BATCH (QUERY_LEN / PREFIX_LEN)

/* calculate when to stop, and which thread has final result */
#define NUM_WAVES                                                              \
  (PREFIX_LEN + (REF_TILE_SIZE - 1) / (SEGMENT_SIZE)) // must be greater than 64
#define WARP_SIZE_MINUS_ONE (WARP_SIZE - 1)
#define RESULT_REG (SEGMENT_SIZE - 1)
#define NUM_WAVES_BY_WARP_SIZE ((NUM_WAVES / WARP_SIZE) * WARP_SIZE)
#define REF_BATCH_MINUS_ONE (REF_BATCH - 1)
#define TWICE_WARP_SIZE_MINUS_ONE ((2 * WARP_SIZE) - 1)

#endif
