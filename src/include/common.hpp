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

#ifdef PINGPONG_BUFFER
#include <cuda/pipeline>
#endif

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

#else
#ifdef PINGPONG_BUFFER
#include <cuda/pipeline>
#endif
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
#endif

#define KMER_LEN 6
#define WARP_SIZE 32
#define SEGMENT_SIZE 32
#define LOG_WARP_SIZE 5
#define QUERY_LEN (4096)
//>=WARP_SIZE for the coalesced shared mem; has to be a multiple of 32; >=64 if
// using PINGPONG buffer

#ifndef FP16
#define REF_LEN                                                                \
  (94 * 1024) // indicates total length of forward + backward squiggle
// genome ; should be a multiple of SEGMENT_SIZE*WARP_SIZE
#else
#define REF_LEN (47 * 1024) // length of fwd strand in case of FP16
#endif

#define BLOCK_NUM (84 * 16)
#define STREAM_NUM 32
#define SMEM_BUFFER_SIZE 1024 // has to be a multiple of 2*WARP_SIZE

#define ADAPTER_LEN 1000
#define ONT_FILE_FORMAT "fast5"

#ifdef PINGPONG_BUFFER

#define STAGES_COUNT 2
#define STAGES_COUNT_MINUS_ONE (STAGES_COUNT - 1)
#define PINGPONG_BUFFER_SIZE (STAGES_COUNT * WARP_SIZE)
// multiple of warp_size
#define PINGPONG_BUFFER_SIZE_MINUS_ONE (PINGPONG_BUFFER_SIZE - 1)

#endif

//-----------------derived variables--------------------------//

#define GROUP_SIZE WARP_SIZE
#define REF_TILE_SIZE (SEGMENT_SIZE * WARP_SIZE)
#define REF_BATCH (REF_LEN / REF_TILE_SIZE)

/* calculate when to stop, and which thread has final result */
#define NUM_WAVES (QUERY_LEN + (REF_TILE_SIZE - 1) / (SEGMENT_SIZE))
#define WARP_SIZE_MINUS_ONE (WARP_SIZE - 1)
#define RESULT_REG (SEGMENT_SIZE - 1)
#define NUM_WAVES_BY_WARP_SIZE ((NUM_WAVES / WARP_SIZE) * WARP_SIZE)
#define REF_BATCH_MINUS_ONE (REF_BATCH - 1)
#define SMEM_BUFFER_SIZE_MINUS_WARP_SIZE (SMEM_BUFFER_SIZE - WARP_SIZE)
#define SMEM_BUFFER_SIZE_MINUS_ONE (SMEM_BUFFER_SIZE - 1)
#define SMEM_BUFFER_PLUS_WARP_SIZE (SMEM_BUFFER_SIZE + WARP_SIZE)
#define SMEM_BUFFER_PLUS_WARP_SIZE_MINUS_ONE (SMEM_BUFFER_SIZE + WARP_SIZE - 1)
#define TWICE_WARP_SIZE (2 * WARP_SIZE)
#define TWICE_WARP_SIZE_MINUS_ONE (TWICE_WARP_SIZE - 1)

#endif