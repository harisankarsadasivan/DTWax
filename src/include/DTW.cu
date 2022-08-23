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
#ifndef FULLDTW
#define FULLDTW

#include "common.hpp"
#include "datatypes.hpp"
#include <cooperative_groups.h>

#ifdef NV_DEBUG
#define REG_ID (SEGMENT_SIZE - 1)
#endif

namespace cg = cooperative_groups;

#define ALL 0xFFFFFFFF
// #define COST_FUNCTION(q, r1, r2, l, t, d)                                      \
 //   FMA(FMA(FMA(r1, FLOAT2HALF2(-1), q), FMA(r1, FLOAT2HALF2(-1), q), 0), r2,      \
 //       FIND_MIN(l, FIND_MIN(t, d)))

#ifndef NO_REF_DEL
#define COST_FUNCTION(q, r1, l, t, d)                                          \
  FMA(FMA(SUB(r1, q), SUB(r1, q), FLOAT2HALF2(0.0f)), FLOAT2HALF2(1.0f),       \
      FIND_MIN(l, FIND_MIN(t, d)))

#else // assuming there are no reference deletions

#define COST_FUNCTION(q, r1, l, t, d)                                          \
  FMA(FMA(SUB(r1, q), SUB(r1, q), FLOAT2HALF2(0.0f)), FLOAT2HALF2(1.0f),       \
      FIND_MIN(t, d))
#endif

// computes segments of the sDTW matrix
template <typename idx_t, typename val_t>
__device__ __forceinline__ void
compute_segment(idxt &wave, val_t &query_val, val_t (&ref_coeff1)[SEGMENT_SIZE],
                val_t &penalty_left, val_t (&penalty_here)[SEGMENT_SIZE],
                val_t &penalty_diag, val_t (&penalty_temp)[2]) {
  /* calculate SEGMENT_SIZE cells */
  penalty_temp[0] = penalty_here[0];

  if (threadIdx.x != (wave - 1)) {
#ifdef NV_DEBUG
#ifndef FP16
    printf(
        "wave= %0d, tid=%0d, query= %0f, ref1= %0f, penalty_here[%0d]= %0f,   \
       penalty_left = % 0f, \
       penalty_diag = % 0f\n",
        wave, threadIdx.x, HALF2FLOAT(query_val),
        HALF2FLOAT(ref_coeff1[REG_ID]), REG_ID,
        HALF2FLOAT(penalty_here[REG_ID]), HALF2FLOAT(penalty_left),
        HALF2FLOAT(penalty_diag));
#else
    printf(
        "wave= %0d, tid=%0d, query= %0f, ref1= %0f, penalty_here[%0d]= %0f,   \
         penalty_left = % 0f, \
         penalty_diag = % 0f\n",
        wave, threadIdx.x, HALF2FLOAT(query_val.x),
        HALF2FLOAT(ref_coeff1[REG_ID].x), REG_ID,
        HALF2FLOAT(penalty_here[REG_ID].x), HALF2FLOAT(penalty_left.x),
        HALF2FLOAT(penalty_diag.x));

#endif
#endif
    penalty_here[0] = COST_FUNCTION(query_val, ref_coeff1[0], penalty_left,
                                    penalty_here[0], penalty_diag);
#if ((SEGMENT_SIZE % 2) == 0)
    for (int i = 1; i < SEGMENT_SIZE - 2; i += 2) {
#else
    for (int i = 1; i < SEGMENT_SIZE - 1; i += 2) {
#endif
      penalty_temp[1] = penalty_here[i];
      penalty_here[i] =
          COST_FUNCTION(query_val, ref_coeff1[i], penalty_here[i - 1],
                        penalty_here[i], penalty_temp[0]);

      penalty_temp[0] = penalty_here[i + 1];
      penalty_here[i + 1] =
          COST_FUNCTION(query_val, ref_coeff1[i + 1], penalty_here[i],
                        penalty_here[i + 1], penalty_temp[1]);
    }
#if ((SEGMENT_SIZE > 1) && ((SEGMENT_SIZE % 2) == 0))
    penalty_here[SEGMENT_SIZE - 1] = COST_FUNCTION(
        query_val, ref_coeff1[SEGMENT_SIZE - 1], penalty_here[SEGMENT_SIZE - 2],
        penalty_here[SEGMENT_SIZE - 1], penalty_temp[0]);
#endif
  }

  else {
    // for (idxt i = 0; i < SEGMENT_SIZE; i++)
    //   penalty_here[i] = FLOAT2HALF2(0);
#ifdef NV_DEBUG
#ifndef FP16
    printf(
        "wave= %0d, tid=%0d, query= %0f, ref1= %0f, penalty_here[%0d]= %0f,   \
       penalty_left = % 0f, \
       penalty_diag = % 0f\n",
        wave, threadIdx.x, HALF2FLOAT(query_val),
        HALF2FLOAT(ref_coeff1[REG_ID]), REG_ID,
        HALF2FLOAT(penalty_here[REG_ID]), HALF2FLOAT(penalty_left),
        HALF2FLOAT(penalty_diag));
#else
    printf(
        "wave= %0d, tid=%0d, query= %0f, ref1= %0f, penalty_here[%0d]= %0f,   \
         penalty_left = % 0f, \
         penalty_diag = % 0f\n",
        wave, threadIdx.x, HALF2FLOAT(query_val.x),
        HALF2FLOAT(ref_coeff1[REG_ID].x), REG_ID,
        HALF2FLOAT(penalty_here[REG_ID].x), HALF2FLOAT(penalty_left.x),
        HALF2FLOAT(penalty_diag.x));
#endif
#endif
    penalty_here[0] = COST_FUNCTION(query_val, ref_coeff1[0], penalty_left,
                                    FLOAT2HALF2(0.0f), penalty_diag);

#if ((SEGMENT_SIZE % 2) == 0)
    for (int i = 1; i < SEGMENT_SIZE - 2; i += 2) {
#else
    for (int i = 1; i < SEGMENT_SIZE - 1; i += 2) {
#endif
      penalty_temp[1] = penalty_here[i];
      penalty_here[i] =
          COST_FUNCTION(query_val, ref_coeff1[i], penalty_here[i - 1],
                        FLOAT2HALF2(0.0f), FLOAT2HALF2(0.0f));

      penalty_temp[0] = penalty_here[i + 1];
      penalty_here[i + 1] =
          COST_FUNCTION(query_val, ref_coeff1[i + 1], penalty_here[i],
                        FLOAT2HALF2(0.0f), FLOAT2HALF2(0.0f));
    }
#if ((SEGMENT_SIZE > 1) && ((SEGMENT_SIZE % 2) == 0))
    penalty_here[SEGMENT_SIZE - 1] = COST_FUNCTION(
        query_val, ref_coeff1[SEGMENT_SIZE - 1], penalty_here[SEGMENT_SIZE - 2],
        FLOAT2HALF2(0.0f), FLOAT2HALF2(0.0f));
#endif
  }
}
//////////////////---------------------------------------------------------------------------------///////////////////////////////////
#ifdef SDTW
/*----------------------------------subsequence
 * DTW--------------------------------*/
template <typename idx_t, typename val_t>
__global__ void DTW(reference_coefficients *ref, val_t *query, val_t *dist,
                    idx_t num_entries, val_t thresh, val_t *penalty_last_col) {

  // cooperative threading
  // cg::thread_block_tile<GROUP_SIZE> g =
  //     cg::tiled_partition<GROUP_SIZE>(cg::this_thread_block());
#if REF_BATCH > 1
  //  __shared__ val_t penalty_here_s[SMEM_BUFFER_SIZE]; ////RBD: have to chnge
  //  this

#ifdef PINGPONG_BUFFER
  __shared__ val_t shared[PINGPONG_BUFFER_SIZE];
  // auto block = cooperative_groups::this_thread_block();
  size_t shared_offset[STAGES_COUNT] = {0, WARP_SIZE};

  // Create a synchronization object (cuda::pipeline)
  cuda::pipeline<cuda::thread_scope_thread> pipeline = cuda::make_pipeline();

#endif

#endif ///!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  // const idx_t base = 0; // blockIdx.x * QUERY_LEN;

  /* initialize penalties */
  val_t penalty_left = FLOAT2HALF2(INFINITY);
  val_t penalty_diag = FLOAT2HALF2(INFINITY);
  val_t penalty_here[SEGMENT_SIZE] = {FLOAT2HALF2(0)};
  val_t penalty_temp[2];
  val_t min_segment = FLOAT2HALF2(INFINITY); // finds min of segment for sDTW
  val_t last_col_penalty_shuffled;           // used to store last col of matrix

  /* each thread computes SEGMENT_SIZE adjacent cells, get corresponding sig
   * values */
  val_t ref_coeff1[SEGMENT_SIZE]; // ref_coeff2[SEGMENT_SIZE];

  /* load next WARP_SIZE query values from memory into new_query_val buffer */
  val_t query_val = FLOAT2HALF2(INFINITY);
  val_t new_query_val = query[blockIdx.x * QUERY_LEN + threadIdx.x];

  /* initialize first thread's chunk */
  if (threadIdx.x == 0) {
    query_val = new_query_val;
  }
  new_query_val = __shfl_down_sync(ALL, new_query_val, 1);
// for (idxt ref_batch = 0; ref_batch < REF_LEN / (REF_TILE_SIZE);
//      ref_batch++) {
#pragma unroll
  for (idxt i = 0; i < SEGMENT_SIZE; i++) {
    ref_coeff1[i] = ref[threadIdx.x + i * WARP_SIZE].coeff1;
    // ref_coeff2[i] = ref[threadIdx.x + i * WARP_SIZE].coeff2;
#ifdef NV_DEBUG
#ifndef FP16
    printf("tid= %0d, ref_coeff1[%0d]=%0f\n", threadIdx.x, i,
           HALF2FLOAT(ref_coeff1[i]));
#else
    printf("tid= %0d, ref_coeff1[%0d]=%0f\n", threadIdx.x, i,
           HALF2FLOAT(ref_coeff1[i].x));
#endif
#endif
  }

  /* calculate full matrix in wavefront parallel manner, multiple cells per
   * thread */
  //#pragma unroll(32)
  for (idxt wave = 1; wave <= NUM_WAVES; wave++) {
    min_segment = __shfl_up_sync((ALL), min_segment, 1);

    compute_segment<idx_t, val_t>(wave, query_val, ref_coeff1, penalty_left,
                                  penalty_here, penalty_diag, penalty_temp);

    /* new_query_val buffer is empty, reload */
    if ((wave & (WARP_SIZE_MINUS_ONE)) == 0) {
      // if (blockIdx.x * QUERY_LEN + wave + threadIdx.x > 32075775)

      new_query_val = query[blockIdx.x * QUERY_LEN + wave + threadIdx.x];
    }

    /* pass next query_value to each thread */
    query_val = __shfl_up_sync(ALL, query_val, 1);

    /* transfer border cell info */

    penalty_diag = penalty_left;

    penalty_left = __shfl_up_sync(ALL, penalty_here[SEGMENT_SIZE - 1], 1);
    if (threadIdx.x == 0) {
      query_val = new_query_val;
      penalty_left = FLOAT2HALF2(INFINITY);
    }
    new_query_val = __shfl_down_sync(ALL, new_query_val, 1);

#if REF_BATCH > 1
    // if ((wave >= WARP_SIZE) && (threadIdx.x == WARP_SIZE_MINUS_ONE)) {
    //   penalty_here_s[(wave - WARP_SIZE)] = penalty_here[RESULT_REG];
    // }
    last_col_penalty_shuffled =
        __shfl_down_sync(ALL, last_col_penalty_shuffled, 1);
    if (threadIdx.x == WARP_SIZE_MINUS_ONE)
      last_col_penalty_shuffled = penalty_here[RESULT_REG];

    //  #if (QUERY_LEN == SMEM_BUFFER_SIZE)
    //      if ((wave >= TWICE_WARP_SIZE)&&((wave & WARP_SIZE_MINUS_ONE) == 0)
    //      && (wave < NUM_WAVES_BY_WARP_SIZE))
    //        penalty_here_s[wave - TWICE_WARP_SIZE + threadIdx.x] =
    //            last_col_penalty_shuffled;
    //      else if ((wave >= NUM_WAVES_BY_WARP_SIZE) &&
    //               (threadIdx.x == WARP_SIZE_MINUS_ONE))
    //        penalty_here_s[wave - WARP_SIZE] = penalty_here[RESULT_REG];

    //  #else

    //      if ((wave >= TWICE_WARP_SIZE)&&((wave & WARP_SIZE_MINUS_ONE) == 0)
    //      &&
    //          (wave < SMEM_BUFFER_SIZE_MINUS_WARP_SIZE))
    //        penalty_here_s[wave - TWICE_WARP_SIZE + threadIdx.x] =
    //            last_col_penalty_shuffled;
    //      else if (((wave & WARP_SIZE_MINUS_ONE) == 0) &&
    //               (wave >=
    //                SMEM_BUFFER_SIZE_MINUS_WARP_SIZE)) // write to global
    //                memory
    //        penalty_last_col[wave - WARP_SIZE + threadIdx.x] =
    //            last_col_penalty_shuffled;

    //  #endif
    // coalesced write to global memory

    if ((wave >= TWICE_WARP_SIZE) && (wave & WARP_SIZE_MINUS_ONE) == 0)
      penalty_last_col[wave - WARP_SIZE + threadIdx.x] =
          last_col_penalty_shuffled;
    else if ((wave >= WARP_SIZE) && (threadIdx.x == WARP_SIZE_MINUS_ONE)) {
      penalty_last_col[wave - WARP_SIZE] = penalty_here[RESULT_REG];
    }

#endif
    // Find min of segment and then shuffle up for sDTW
    if (wave >= QUERY_LEN) {
      for (idxt i = 0; i < SEGMENT_SIZE; i++) {
        min_segment = FIND_MIN(min_segment, penalty_here[i]);
      }
#ifdef NV_DEBUG
      if (threadIdx.x == (wave - QUERY_LEN))
        printf("minsegment=%0f,wave=%0d, refbatch=0, tid=%0d\n", min_segment,
               wave, threadIdx.x);
#endif
    }
  }

  /*------------------------------for all ref batches > 0
   * ---------------------------------- */
#if REF_BATCH > 1
  for (idxt ref_batch = 1; ref_batch < REF_BATCH - 1; ref_batch++) {
#ifdef NV_DEBUG
    printf("refbatch=%0d\n", ref_batch);
#endif
    min_segment = __shfl_down_sync((ALL), min_segment, 31);

    /* initialize penalties */
    // val_t penalty_left = FLOAT2HALF2(INFINITY);
    penalty_diag = FLOAT2HALF2(INFINITY);
    penalty_left = FLOAT2HALF2(INFINITY);
    for (auto i = 0; i < SEGMENT_SIZE; i++)
      penalty_here[i] = FLOAT2HALF2(0);
    // for (auto i = 0; i < 2; i++)
    //   penalty_temp[i] = FLOAT2HALF2(INFINITY);

    /* load next WARP_SIZE query values from memory into new_query_val buffer */
    query_val = FLOAT2HALF2(INFINITY);
    new_query_val = query[blockIdx.x * QUERY_LEN + threadIdx.x];

    /* initialize first thread's chunk */
    if (threadIdx.x == 0) {
      query_val = new_query_val;

      penalty_left = penalty_last_col[0];
    }

    new_query_val = __shfl_down_sync(ALL, new_query_val, 1);

    for (idxt i = 0; i < SEGMENT_SIZE; i++) {
      ref_coeff1[i] =
          ref[ref_batch * (REF_TILE_SIZE) + threadIdx.x + i * WARP_SIZE].coeff1;
      // ref_coeff2[i] =
      //   ref[ref_batch * (REF_TILE_SIZE) + threadIdx.x + i *
      //   WARP_SIZE].coeff2;
    }
    /* calculate full matrix in wavefront parallel manner, multiple cells per
     * thread */
    //#pragma unroll(32)
    for (idxt wave = 1; wave <= NUM_WAVES; wave++) {

      compute_segment<idx_t, val_t>(wave, query_val, ref_coeff1, penalty_left,
                                    penalty_here, penalty_diag, penalty_temp);

      /* new_query_val buffer is empty, reload */
      if ((wave & (WARP_SIZE_MINUS_ONE)) == 0) {
        new_query_val = query[blockIdx.x * QUERY_LEN + wave + threadIdx.x];
      }

      /* pass next query_value to each thread */
      query_val = __shfl_up_sync(ALL, query_val, 1);
      if (threadIdx.x == 0) {
        query_val = new_query_val;
      }

      last_col_penalty_shuffled =
          __shfl_down_sync(ALL, last_col_penalty_shuffled, 1);
      if (threadIdx.x == WARP_SIZE_MINUS_ONE)
        last_col_penalty_shuffled = penalty_here[RESULT_REG];
      //  #if (QUERY_LEN == SMEM_BUFFER_SIZE)
      //        if ((wave >= TWICE_WARP_SIZE)&&((wave & WARP_SIZE_MINUS_ONE) ==
      //        0) &&
      //            (wave < NUM_WAVES_BY_WARP_SIZE))
      //          penalty_here_s[(wave - WARP_SIZE) + threadIdx.x] =
      //              last_col_penalty_shuffled;
      //        else if ((wave >= NUM_WAVES_BY_WARP_SIZE) &&
      //                 (threadIdx.x == WARP_SIZE_MINUS_ONE))
      //          penalty_here_s[(wave - WARP_SIZE)] = penalty_here[RESULT_REG];
      //  #else
      //        if ((wave >= TWICE_WARP_SIZE)&&((wave & WARP_SIZE_MINUS_ONE) ==
      //        0) &&
      //            (wave < SMEM_BUFFER_SIZE_MINUS_WARP_SIZE))
      //          penalty_here_s[(wave - WARP_SIZE) + threadIdx.x] =
      //              last_col_penalty_shuffled;
      //        else if (((wave & WARP_SIZE_MINUS_ONE) == 0) &&
      //                 (wave >=
      //                  SMEM_BUFFER_SIZE_MINUS_WARP_SIZE)) // write to global
      //                  memory
      //          penalty_last_col[(wave - WARP_SIZE) + threadIdx.x] =
      //              last_col_penalty_shuffled;

      //  #endif
      // coalesced write to global memory
      if ((wave >= TWICE_WARP_SIZE) && (wave & WARP_SIZE_MINUS_ONE) == 0)
        penalty_last_col[wave - WARP_SIZE + threadIdx.x] =
            last_col_penalty_shuffled;
      else if ((wave >= WARP_SIZE) && (threadIdx.x == WARP_SIZE_MINUS_ONE)) {
        penalty_last_col[wave - WARP_SIZE] = penalty_here[RESULT_REG];
      }

      new_query_val = __shfl_down_sync(ALL, new_query_val, 1);

      /* transfer border cell info */
      penalty_diag = penalty_left;
      penalty_left = __shfl_up_sync(ALL, penalty_here[SEGMENT_SIZE - 1], 1);

#ifdef PINGPONG_BUFFER
      if ((wave > SMEM_BUFFER_SIZE_MINUS_ONE) &&
          ((wave % PINGPONG_BUFFER_SIZE) == 0)) {
        for (size_t fetch = wave; fetch < (wave + STAGES_COUNT); ++fetch) {
          pipeline.producer_acquire();
          cuda::memcpy_async(
              cooperative_groups::this_thread(),
              (float *)(shared) + shared_offset[fetch % 2] + threadIdx.x,
              (float *)(penalty_last_col) + wave + threadIdx.x,
              cuda::aligned_size_t<4>(sizeof(val_t) * WARP_SIZE), pipeline);
          pipeline.producer_commit(); // Commit the fetch-ahead stage
        }
        // printf("after pipeline prod commit, wave=%0d\n", wave);
      }

#endif

#ifdef PINGPONG_BUFFER
      pipeline.consumer_wait(); // Wait for ‘subset’ stage tobeavailable
      // printf("after pipeline cons wait, wave=%0d\n", wave);
      __syncwarp();
#endif

      if (threadIdx.x == 0) {
        penalty_left = penalty_last_col[wave];
        //  #if (QUERY_LEN == SMEM_BUFFER_SIZE)
        //          penalty_left = penalty_here_s[wave];
        //  #else

        //  #ifdef PINGPONG_BUFFER
        //          if (wave <= SMEM_BUFFER_SIZE_MINUS_ONE)
        //            penalty_left = penalty_here_s[wave];
        //          else {

        //            penalty_left = shared[(wave &
        //            PINGPONG_BUFFER_SIZE_MINUS_ONE)];
        //          }
        //  #else
        //          if (wave <= SMEM_BUFFER_SIZE_MINUS_ONE)
        //            penalty_left = penalty_here_s[wave];
        //          else
        //            penalty_left = penalty_last_col[wave];
        //  #endif
        //  #endif
      }

#ifdef PINGPONG_BUFFER
      pipeline.consumer_release();
      // printf("after cons release wave=%0d\n", wave);
#endif

      // Find min of segment and then shuffle up for sDTW
      if (wave >= QUERY_LEN) {
        for (idxt i = 0; i < SEGMENT_SIZE; i++) {
          min_segment = FIND_MIN(min_segment, penalty_here[i]);
        }
#ifdef NV_DEBUG
        if (threadIdx.x == (wave - QUERY_LEN))
          printf("minsegment=%0f,tid=%0d,wave=%0d, refbatch=%0d\n", min_segment,
                 threadIdx.x, wave, ref_batch);
#endif
        if (wave != (NUM_WAVES))
          min_segment = __shfl_up_sync((ALL), min_segment, 1);
      }
    }
  }

//----------------------------last
// sub-matrix calculation or
// ref_batch=REF_BATCH-1------------------------------------------------//
#ifdef NV_DEBUG
  printf("refbatch=%0d\n", REF_BATCH_MINUS_ONE);
#endif
  min_segment = __shfl_down_sync((ALL), min_segment, 31);

  /* initialize penalties */
  // val_t penalty_left = FLOAT2HALF2(INFINITY);
  penalty_diag = FLOAT2HALF2(INFINITY);
  penalty_left = FLOAT2HALF2(INFINITY);
  for (auto i = 0; i < SEGMENT_SIZE; i++)
    penalty_here[i] = FLOAT2HALF2(0);
  // for (auto i = 0; i < 2; i++)
  //   penalty_temp[i] = FLOAT2HALF2(INFINITY);

  /* load next WARP_SIZE query values from memory into new_query_val buffer */
  query_val = FLOAT2HALF2(INFINITY);
  new_query_val = query[blockIdx.x * QUERY_LEN + threadIdx.x];

  /* initialize first thread's chunk */
  if (threadIdx.x == 0) {
    query_val = new_query_val;
    penalty_left = penalty_last_col[0];
    //  penalty_left = penalty_here_s[0];
  }

  new_query_val = __shfl_down_sync(ALL, new_query_val, 1);

  for (idxt i = 0; i < SEGMENT_SIZE; i++) {
    ref_coeff1[i] =
        ref[REF_BATCH_MINUS_ONE * (REF_TILE_SIZE) + threadIdx.x + i * WARP_SIZE]
            .coeff1;
    // ref_coeff2[i] =
    //   ref[ref_batch * (REF_TILE_SIZE) + threadIdx.x + i * WARP_SIZE].coeff2;
  }
  /* calculate full matrix in wavefront parallel manner, multiple cells per
   * thread */
  //#pragma unroll(32)
  for (idxt wave = 1; wave <= NUM_WAVES; wave++) {

    compute_segment<idx_t, val_t>(wave, query_val, ref_coeff1, penalty_left,
                                  penalty_here, penalty_diag, penalty_temp);

    /* new_query_val buffer is empty, reload */
    if ((wave & (WARP_SIZE_MINUS_ONE)) == 0) {
      new_query_val = query[blockIdx.x * QUERY_LEN + wave + threadIdx.x];
    }

    /* pass next query_value to each thread */
    query_val = __shfl_up_sync(ALL, query_val, 1);
    if (threadIdx.x == 0) {
      query_val = new_query_val;
    }

    new_query_val = __shfl_down_sync(ALL, new_query_val, 1);

    /* transfer border cell info */
    penalty_diag = penalty_left;
    penalty_left = __shfl_up_sync(ALL, penalty_here[SEGMENT_SIZE - 1], 1);

    if (threadIdx.x == 0) {
      penalty_left = penalty_last_col[wave];
      //  #if (QUERY_LEN == SMEM_BUFFER_SIZE)
      //        penalty_left = penalty_here_s[wave];
      //  #else
      //        if (wave <= SMEM_BUFFER_SIZE_MINUS_ONE)
      //          penalty_left = penalty_here_s[wave];
      //        else
      //          penalty_left = penalty_last_col[wave];
      //  #endif
    }

    // Find min of segment and then shuffle up for sDTW
    if (wave >= QUERY_LEN) {
      for (idxt i = 0; i < SEGMENT_SIZE; i++) {
        min_segment = FIND_MIN(min_segment, penalty_here[i]);
      }
#ifdef NV_DEBUG
      if (threadIdx.x == (wave - QUERY_LEN))
        printf("minsegment=%0f,tid=%0d,wave=%0d,ref_batch=%0d\n", min_segment,
               threadIdx.x, wave, REF_BATCH_MINUS_ONE);
#endif

      if (wave != (NUM_WAVES))
        min_segment = __shfl_up_sync((ALL), min_segment, 1);
    }
  }

#endif
  //-------------------------------------------------------------------------------------------------------------------//
  /* return result */
  if (threadIdx.x == WARP_SIZE_MINUS_ONE) {

    dist[blockIdx.x] = min_segment;

#ifdef NV_DEBUG
    printf("min_segment=%0f, tid=%0d, blockid=%0d\n", min_segment, threadIdx.x,
           blockIdx.x);
#endif
  }
  return;
}
/////////////////////////
///////////////////////////
///////////////////-----------------
#else // DTW
// /*template <typename idx_t, typename val_t>
// __global__ void DTW(reference_coefficients *ref, val_t *query, val_t *dist,
//                     idx_t num_entries, val_t thresh) {

//   // cooperative threading
//   cg::thread_block_tile<GROUP_SIZE> g =
//       cg::tiled_partition<GROUP_SIZE>(cg::this_thread_block());

//   __shared__ val_t
//       penalty_here_s[QUERY_LEN]; ////RBD: have to chnge this
//                                  ///!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

//   /* create vars for indexing */
//   const idx_t blockIdx.x = blockIdx.x;
//   const idx_t threadIdx.x = cg::this_thread_block().thread_rank();
//   // const idx_t base = 0; // blockIdx.x * QUERY_LEN;

//   /* initialize penalties */
//   val_t penalty_left = FLOAT2HALF2(INFINITY);
//   val_t penalty_diag = FLOAT2HALF2(INFINITY);
//   val_t penalty_here[SEGMENT_SIZE] = {FLOAT2HALF2(INFINITY)};
//   val_t penalty_temp[2];

//   /* each thread computes SEGMENT_SIZE adjacent cells, get corresponding
//   sig
//    * values */
//   val_t ref_coeff1[SEGMENT_SIZE], ref_coeff2[SEGMENT_SIZE];

//   /* load next WARP_SIZE query values from memory into new_query_val buffer
//   */ val_t query_val = FLOAT2HALF2(INFINITY); val_t new_query_val =
//   query[blockIdx.x * QUERY_LEN + threadIdx.x];

//   /* initialize first thread's chunk */
//   if (threadIdx.x == 0) {
//     query_val = new_query_val;
//     penalty_diag = FLOAT2HALF2(0);
//   }
//   new_query_val = __shfl_down_sync(ALL, new_query_val, 1);
//   // for (idxt ref_batch = 0; ref_batch < REF_LEN / (REF_TILE_SIZE);
//   //      ref_batch++) {
//   for (idxt i = 0; i < SEGMENT_SIZE; i++) {
//     // ref_coeff1[i] = ref[SEGMENT_SIZE * threadIdx.x + i];

//     ref_coeff1[i] = ref[threadIdx.x + i * WARP_SIZE].coeff1;
//     ref_coeff2[i] = ref[threadIdx.x + i * WARP_SIZE].coeff2;
//     printf("ref_coeff1[%0d]=%0f\n", i, HALF2FLOAT(ref_coeff1[i]));
//   }

//   /* calculate full matrix in wavefront parallel manner, multiple cells per
//    * thread */
//   for (idxt wave = 1; wave <= NUM_WAVES; wave++) {

//     /* calculate SEGMENT_SIZE cells */
//     penalty_temp[0] = penalty_here[0];
//     penalty_here[0] =
//         COST_FUNCTION(query_val, ref_coeff1[0], ref_coeff2[0],
//         penalty_left,
//                       penalty_here[0], penalty_diag);

//     for (int i = 1; i < SEGMENT_SIZE - 2; i += 2) {
//       penalty_temp[1] = penalty_here[i];
//       penalty_here[i] =
//           COST_FUNCTION(query_val, ref_coeff1[i], ref_coeff2[i],
//                         penalty_here[i - 1], penalty_here[i],
//                         penalty_temp[0]);

//       penalty_temp[0] = penalty_here[i + 1];
//       penalty_here[i + 1] =
//           COST_FUNCTION(query_val, ref_coeff1[i + 1], ref_coeff2[i + 1],
//                         penalty_here[i], penalty_here[i + 1],
//                         penalty_temp[1]);
//     }
// #ifndef NV_DEBUG
//     penalty_here[SEGMENT_SIZE - 1] = COST_FUNCTION(
//         query_val, ref_coeff1[SEGMENT_SIZE - 1], ref_coeff2[SEGMENT_SIZE -
//         1], penalty_here[SEGMENT_SIZE - 2], penalty_here[SEGMENT_SIZE - 1],
//         penalty_temp[0]);

// #else
//     penalty_here[SEGMENT_SIZE - 1] = COST_FUNCTION(
//         query_val, ref_coeff1[SEGMENT_SIZE - 1], ref_coeff2[SEGMENT_SIZE -
//         1], FLOAT2HALF2(INFINITY), penalty_here[SEGMENT_SIZE - 1],
//         penalty_temp[0]);
// #endif

//     /* new_query_val buffer is empty, reload */
//     if ((wave & (WARP_SIZE_MINUS_ONE)) == 0) {
//       new_query_val = query[blockIdx.x * QUERY_LEN + wave + threadIdx.x];
//     }

//     /* pass next query_value to each thread */
//     query_val = __shfl_up_sync(ALL, query_val, 1);
//     if (threadIdx.x == 0) {
//       query_val = new_query_val;
//     }
//     new_query_val = __shfl_down_sync(ALL, new_query_val, 1);

//     /* transfer border cell info */
//     penalty_diag = penalty_left;
//     penalty_left = __shfl_up_sync(ALL, penalty_here[SEGMENT_SIZE - 1], 1);
//     if (threadIdx.x == 0) {
//       penalty_left = FLOAT2HALF2(INFINITY);
//     }
//     if ((wave >= WARP_SIZE) && (threadIdx.x == WARP_SIZE_MINUS_ONE)) {
//       penalty_here_s[(wave - WARP_SIZE)] = penalty_here[RESULT_REG];
//     }
//   }

//   printf("final_score=%0f\n", HALF2FLOAT(penalty_here[RESULT_REG]));
//   /* return result */
//   if ((threadIdx.x == WARP_SIZE_MINUS_ONE) && (REF_BATCH == 1)) {
//     // printf("@@@result_threadId=%0d\n",WARP_SIZE_MINUS_ONE);

//     dist[blockIdx.x] =
//         penalty_here[RESULT_REG] > thresh ? FLOAT2HALF2(0) :
//         FLOAT2HALF2(1);
//     return;
//   }

//   /*------------------------------for all ref batches > 0
//    * ---------------------------------- */
//   for (idxt ref_batch = 1; ref_batch < REF_BATCH; ref_batch++) {
//     /* initialize penalties */
//     penalty_left = FLOAT2HALF2(INFINITY);
//     penalty_diag = FLOAT2HALF2(INFINITY);
//     for (auto i = 0; i < SEGMENT_SIZE; i++)
//       penalty_here[i] = FLOAT2HALF2(INFINITY);
//     for (auto i = 0; i < 2; i++)
//       penalty_temp[i] = FLOAT2HALF2(INFINITY);

//     /* load next WARP_SIZE query values from memory into new_query_val
//     buffer
//     */ query_val = FLOAT2HALF2(INFINITY); new_query_val = query[blockIdx.x
//     * QUERY_LEN +
//                           QUERY_LEN * (ref_batch) / REF_BATCH +
//                           threadIdx.x];

//     /* initialize first thread's chunk */
//     if (threadIdx.x == 0) {
//       query_val = new_query_val;
//       penalty_diag = FLOAT2HALF2(0);
//       penalty_left = penalty_here_s[0];
//     }
//     new_query_val = __shfl_down_sync(ALL, new_query_val, 1);

//     for (idxt i = 0; i < SEGMENT_SIZE; i++) {
//       ref_coeff1[i] =
//           ref[ref_batch * (REF_TILE_SIZE) + SEGMENT_SIZE * threadIdx.x + i]
//               .coeff1;
//       ref_coeff2[i] =
//           ref[ref_batch * (REF_TILE_SIZE) + SEGMENT_SIZE * threadIdx.x + i]
//               .coeff2;
//     }
//     /* calculate full matrix in wavefront parallel manner, multiple cells
//     per
//      * thread */
//     for (idxt wave = 1; wave <= NUM_WAVES; wave++) {

//       /* calculate SEGMENT_SIZE cells */
//       penalty_temp[0] = penalty_here[0];
//       penalty_here[0] =
//           COST_FUNCTION(query_val, ref_coeff1[0], ref_coeff2[0],
//           penalty_left,
//                         penalty_here[0], penalty_diag);

//       for (int i = 1; i < SEGMENT_SIZE - 2; i += 2) {
//         penalty_temp[1] = penalty_here[i];
//         penalty_here[i] = COST_FUNCTION(query_val, ref_coeff1[i],
//         ref_coeff2[i],
//                                         penalty_here[i - 1],
//                                         penalty_here[i], penalty_temp[0]);

//         penalty_temp[0] = penalty_here[i + 1];
//         penalty_here[i + 1] = COST_FUNCTION(
//             query_val, ref_coeff1[i + 1], ref_coeff2[i + 1],
//             penalty_here[i], penalty_here[i + 1], penalty_temp[1]);
//       }
// #ifndef NV_DEBUG
//       penalty_here[SEGMENT_SIZE - 1] = COST_FUNCTION(
//           query_val, ref_coeff1[SEGMENT_SIZE - 1], ref_coeff2[SEGMENT_SIZE
//           - 1], penalty_here[SEGMENT_SIZE - 2], penalty_here[SEGMENT_SIZE -
//           1], penalty_temp[0]);

// #else
//       penalty_here[SEGMENT_SIZE - 1] =
//           COST_FUNCTION(query_val, ref_coeff1[SEGMENT_SIZE - 1],
//                         ref_coeff2[SEGMENT_SIZE - 1],
//                         FLOAT2HALF2(INFINITY), penalty_here[SEGMENT_SIZE -
//                         1], penalty_temp[0]);
// #endif

//       /* new_query_val buffer is empty, reload */
//       if ((wave & (WARP_SIZE_MINUS_ONE)) == 0) {
//         new_query_val = query[blockIdx.x * QUERY_LEN + wave + threadIdx.x];
//       }

//       /* pass next query_value to each thread */
//       query_val = __shfl_up_sync(ALL, query_val, 1);
//       if (threadIdx.x == 0) {
//         query_val = new_query_val;
//       }
//       new_query_val = __shfl_down_sync(ALL, new_query_val, 1);

//       /* transfer border cell info */
//       penalty_diag = penalty_left;
//       penalty_left = __shfl_up_sync(ALL, penalty_here[SEGMENT_SIZE - 1],
//       1); if (threadIdx.x == 0) {
//         penalty_left = penalty_here_s[wave];
//       }
//       if ((wave >= WARP_SIZE) && (threadIdx.x == WARP_SIZE_MINUS_ONE)) {
//         penalty_here_s[(wave - WARP_SIZE)] = penalty_here[RESULT_REG];
//       }
//     }
//     /* return result */
//     if ((threadIdx.x == WARP_SIZE_MINUS_ONE) && (ref_batch == (REF_BATCH -
//     1)))
//     {
//       // printf("@@@result_threadId=%0d\n",WARP_SIZE_MINUS_ONE);

//       dist[blockIdx.x] =
//           penalty_here[RESULT_REG] > thresh ? FLOAT2HALF2(0) :
//           FLOAT2HALF2(1);

//       return;
//     }
//   }
// }

#endif //} //DTW

#endif //}