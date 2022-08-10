#ifndef FULLDTW
#define FULLDTW

#include "common.hpp"
#include "datatypes.hpp"
#include <cooperative_groups.h>

#ifdef NV_DEBUG
#define REG_ID (SEGMENT_SIZE - 1)
#endif

#ifdef FP16 // FP16 definitions
#include <cuda_fp16.h>
#endif

namespace cg = cooperative_groups;

#define ALL 0xFFFFFFFF
#define COST_FUNCTION(q, r1, r2, l, t, d)                                      \
  FMA(FMA(FMA(r1, FLOAT2HALF(-1), q), FMA(r1, FLOAT2HALF(-1), q), 0), r2,      \
      FIND_MIN(l, FIND_MIN(t, d)))

// computes segments of the sDTW matrix
template <typename idx_t, typename val_t>
__device__ __forceinline__ void
compute_segment(idxt &wave, const idx_t &thread_id, val_t &query_val,
                val_t (&ref_coeff1)[SEGMENT_SIZE],
                val_t (&ref_coeff2)[SEGMENT_SIZE], val_t &penalty_left,
                val_t (&penalty_here)[SEGMENT_SIZE], val_t &penalty_diag,
                val_t (&penalty_temp)[2]) {
  /* calculate SEGMENT_SIZE cells */
  penalty_temp[0] = penalty_here[0];

  if (thread_id != (wave - 1)) {
#ifdef NV_DEBUG
    printf(
        "wave= %0d, tid=%0d, query= %0f, ref1= %0f, ref2= %0f, penalty_here[%0d]= %0f,   \
      penalty_left = % 0f, \
      penalty_diag = % 0f\n",
        wave, thread_id, HALF2FLOAT(query_val), HALF2FLOAT(ref_coeff1[REG_ID]),
        HALF2FLOAT(ref_coeff2[REG_ID]), REG_ID,
        HALF2FLOAT(penalty_here[REG_ID]), HALF2FLOAT(penalty_left),
        HALF2FLOAT(penalty_diag));
#endif
    penalty_here[0] =
        COST_FUNCTION(query_val, ref_coeff1[0], ref_coeff2[0], penalty_left,
                      penalty_here[0], penalty_diag);
#if ((SEGMENT_SIZE % 2) == 0)
    for (int i = 1; i < SEGMENT_SIZE - 2; i += 2) {
#else
    for (int i = 1; i < SEGMENT_SIZE - 1; i += 2) {
#endif
      penalty_temp[1] = penalty_here[i];
      penalty_here[i] =
          COST_FUNCTION(query_val, ref_coeff1[i], ref_coeff2[i],
                        penalty_here[i - 1], penalty_here[i], penalty_temp[0]);

      penalty_temp[0] = penalty_here[i + 1];
      penalty_here[i + 1] =
          COST_FUNCTION(query_val, ref_coeff1[i + 1], ref_coeff2[i + 1],
                        penalty_here[i], penalty_here[i + 1], penalty_temp[1]);
    }
#if ((SEGMENT_SIZE > 1) && ((SEGMENT_SIZE % 2) == 0))
    penalty_here[SEGMENT_SIZE - 1] = COST_FUNCTION(
        query_val, ref_coeff1[SEGMENT_SIZE - 1], ref_coeff2[SEGMENT_SIZE - 1],
        penalty_here[SEGMENT_SIZE - 2], penalty_here[SEGMENT_SIZE - 1],
        penalty_temp[0]);
#endif
  }

  else {
    // for (idxt i = 0; i < SEGMENT_SIZE; i++)
    //   penalty_here[i] = FLOAT2HALF(0);
#ifdef NV_DEBUG
    printf(
        "wave= %0d, tid=%0d, query= %0f, ref1= %0f, ref2= %0f, penalty_here[%0d]= %0f,   \
      penalty_left = % 0f, \
      penalty_diag = % 0f\n",
        wave, thread_id, HALF2FLOAT(query_val), HALF2FLOAT(ref_coeff1[REG_ID]),
        HALF2FLOAT(ref_coeff2[REG_ID]), REG_ID,
        HALF2FLOAT(penalty_here[REG_ID]), HALF2FLOAT(penalty_left),
        HALF2FLOAT(penalty_diag));
#endif
    penalty_here[0] =
        COST_FUNCTION(query_val, ref_coeff1[0], ref_coeff2[0], penalty_left,
                      FLOAT2HALF(0.0f), penalty_diag);

#if ((SEGMENT_SIZE % 2) == 0)
    for (int i = 1; i < SEGMENT_SIZE - 2; i += 2) {
#else
    for (int i = 1; i < SEGMENT_SIZE - 1; i += 2) {
#endif
      penalty_temp[1] = penalty_here[i];
      penalty_here[i] = COST_FUNCTION(query_val, ref_coeff1[i], ref_coeff2[i],
                                      penalty_here[i - 1], FLOAT2HALF(0.0f),
                                      FLOAT2HALF(0.0f));

      penalty_temp[0] = penalty_here[i + 1];
      penalty_here[i + 1] =
          COST_FUNCTION(query_val, ref_coeff1[i + 1], ref_coeff2[i + 1],
                        penalty_here[i], FLOAT2HALF(0.0f), FLOAT2HALF(0.0f));
    }
#if ((SEGMENT_SIZE > 1) && ((SEGMENT_SIZE % 2) == 0))
    penalty_here[SEGMENT_SIZE - 1] = COST_FUNCTION(
        query_val, ref_coeff1[SEGMENT_SIZE - 1], ref_coeff2[SEGMENT_SIZE - 1],
        penalty_here[SEGMENT_SIZE - 2], FLOAT2HALF(0.0f), FLOAT2HALF(0.0f));
#endif
  }
}
//////////////////---------------------------------------------------------------------------------///////////////////////////////////
#ifdef SDTW
/*----------------------------------subsequence
 * DTW--------------------------------*/
template <typename idx_t, typename val_t>
__global__ void DTW(reference_coefficients *ref, val_t *query, val_t *dist,
                    idx_t num_entries, val_t thresh) {

  // cooperative threading
  cg::thread_block_tile<GROUP_SIZE> g =
      cg::tiled_partition<GROUP_SIZE>(cg::this_thread_block());

#if REF_BATCH > 1
  __shared__ val_t penalty_here_s[QUERY_LEN]; ////RBD: have to chnge this
#endif ///!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  /* create vars for indexing */
  const idx_t block_id = blockIdx.x;
  const idx_t thread_id = cg::this_thread_block().thread_rank();
  // const idx_t base = 0; // block_id * QUERY_LEN;

  /* initialize penalties */
  val_t penalty_left = FLOAT2HALF(INFINITY);
  val_t penalty_diag = FLOAT2HALF(INFINITY);
  val_t penalty_here[SEGMENT_SIZE] = {FLOAT2HALF(0)};
  val_t penalty_temp[2];
  val_t min_segment = FLOAT2HALF(INFINITY); // finds min of segment for sDTW

  /* each thread computes SEGMENT_SIZE adjacent cells, get corresponding sig
   * values */
  val_t ref_coeff1[SEGMENT_SIZE], ref_coeff2[SEGMENT_SIZE];

  /* load next WARP_SIZE query values from memory into new_query_val buffer */
  val_t query_val = FLOAT2HALF(INFINITY);
  val_t new_query_val = query[block_id * QUERY_LEN + thread_id];

  /* initialize first thread's chunk */
  if (thread_id == 0) {
    query_val = new_query_val;
  }
  new_query_val = __shfl_down_sync(ALL, new_query_val, 1);
// for (idxt ref_batch = 0; ref_batch < REF_LEN / (REF_TILE_SIZE);
//      ref_batch++) {
#pragma unroll
  for (idxt i = 0; i < SEGMENT_SIZE; i++) {
    ref_coeff1[i] = ref[thread_id + i * WARP_SIZE].coeff1;
    ref_coeff2[i] = ref[thread_id + i * WARP_SIZE].coeff2;
#ifdef NV_DEBUG
    printf("tid= %0d, ref_coeff1[%0d]=%0f\n", thread_id, i,
           HALF2FLOAT(ref_coeff1[i]));
#endif
  }

  /* calculate full matrix in wavefront parallel manner, multiple cells per
   * thread */
  for (idxt wave = 1; wave <= NUM_WAVES; wave++) {
    min_segment = __shfl_up_sync((ALL), min_segment, 1);

    compute_segment<idx_t, val_t>(wave, thread_id, query_val, ref_coeff1,
                                  ref_coeff2, penalty_left, penalty_here,
                                  penalty_diag, penalty_temp);

    /* new_query_val buffer is empty, reload */
    if ((wave & (WARP_SIZE - 1)) == 0) {
      new_query_val = query[block_id * QUERY_LEN + wave + thread_id];
    }

    /* pass next query_value to each thread */
    query_val = __shfl_up_sync(ALL, query_val, 1);
    if (thread_id == 0) {
      query_val = new_query_val;
    }
    new_query_val = __shfl_down_sync(ALL, new_query_val, 1);

    /* transfer border cell info */

    penalty_diag = penalty_left;

    penalty_left = __shfl_up_sync(ALL, penalty_here[SEGMENT_SIZE - 1], 1);
    if (thread_id == 0) {
      penalty_left = FLOAT2HALF(INFINITY);
    }

#if REF_BATCH > 1
    if ((wave >= WARP_SIZE) && (thread_id == RESULT_THREAD_ID)) {
      penalty_here_s[(wave - WARP_SIZE)] = penalty_here[RESULT_REG];
    }
#endif
    // Find min of segment and then shuffle up for sDTW
    if (wave >= QUERY_LEN) {
      for (idxt i = 0; i < SEGMENT_SIZE; i++) {
        min_segment = FIND_MIN(min_segment, penalty_here[i]);
      }
    }
  }

  /* return result */

  if ((thread_id == RESULT_THREAD_ID) && (REF_BATCH == 1)) {

    dist[block_id] = HALF2FLOAT(min_segment);
#ifdef NV_DEBUG
    printf("min_segment=%0f, tid=%0d, blockid=%0d\n", min_segment, thread_id,
           block_id);
#endif
    return;
  }

  /*------------------------------for all ref batches > 0
   * ---------------------------------- */
  for (idxt ref_batch = 1; ref_batch < REF_BATCH; ref_batch++) {

    min_segment = __shfl_down_sync((ALL), min_segment, 31);

    /* initialize penalties */
    // val_t penalty_left = FLOAT2HALF(INFINITY);
    penalty_diag = FLOAT2HALF(INFINITY);
    penalty_left = FLOAT2HALF(INFINITY);
    for (auto i = 0; i < SEGMENT_SIZE; i++)
      penalty_here[i] = FLOAT2HALF(0);
    // for (auto i = 0; i < 2; i++)
    //   penalty_temp[i] = FLOAT2HALF(INFINITY);

    /* load next WARP_SIZE query values from memory into new_query_val buffer */
    query_val = FLOAT2HALF(INFINITY);
    new_query_val = query[block_id * QUERY_LEN + thread_id];

    /* initialize first thread's chunk */
    if (thread_id == 0) {
      query_val = new_query_val;
#if REF_BATCH > 1
      penalty_left = penalty_here_s[0];
#endif
    }
    new_query_val = __shfl_down_sync(ALL, new_query_val, 1);

    for (idxt i = 0; i < SEGMENT_SIZE; i++) {
      ref_coeff1[i] =
          ref[ref_batch * (REF_TILE_SIZE) + thread_id + i * WARP_SIZE].coeff1;
      ref_coeff2[i] =
          ref[ref_batch * (REF_TILE_SIZE) + thread_id + i * WARP_SIZE].coeff2;
    }
    /* calculate full matrix in wavefront parallel manner, multiple cells per
     * thread */
    for (idxt wave = 1; wave <= NUM_WAVES; wave++) {

      compute_segment<idx_t, val_t>(wave, thread_id, query_val, ref_coeff1,
                                    ref_coeff2, penalty_left, penalty_here,
                                    penalty_diag, penalty_temp);

      /* new_query_val buffer is empty, reload */
      if ((wave & (WARP_SIZE - 1)) == 0) {
        new_query_val = query[block_id * QUERY_LEN + wave + thread_id];
      }

      /* pass next query_value to each thread */
      query_val = __shfl_up_sync(ALL, query_val, 1);
      if (thread_id == 0) {
        query_val = new_query_val;
      }
      new_query_val = __shfl_down_sync(ALL, new_query_val, 1);

      /* transfer border cell info */
      penalty_diag = penalty_left;
      penalty_left = __shfl_up_sync(ALL, penalty_here[SEGMENT_SIZE - 1], 1);

#if REF_BATCH > 1
      if (thread_id == 0) {

        penalty_left = penalty_here_s[wave];
      } else if ((wave >= WARP_SIZE) && (thread_id == RESULT_THREAD_ID) &&
                 (ref_batch < (REF_BATCH - 1))) {
        penalty_here_s[(wave - WARP_SIZE)] = penalty_here[RESULT_REG];

      }
#endif
      // Find min of segment and then shuffle up for sDTW
      if (wave >= QUERY_LEN) {
        for (idxt i = 0; i < SEGMENT_SIZE; i++) {
          min_segment = FIND_MIN(min_segment, penalty_here[i]);
        }
        if (wave != (NUM_WAVES))
          min_segment = __shfl_up_sync((ALL), min_segment, 1);
      }
    }
    /* return result */
    if ((thread_id == RESULT_THREAD_ID) && (ref_batch == (REF_BATCH - 1))) {

      dist[block_id] = HALF2FLOAT(min_segment);

#ifdef NV_DEBUG
      printf("min_segment=%0f, tid=%0d, blockid=%0d\n", min_segment, thread_id,
             block_id);
#endif

      return;
    }
  }
}
/////////////////////////
///////////////////////////
///////////////////-----------------
#else // DTW
template <typename idx_t, typename val_t>
__global__ void DTW(reference_coefficients *ref, val_t *query, val_t *dist,
                    idx_t num_entries, val_t thresh) {

  // cooperative threading
  cg::thread_block_tile<GROUP_SIZE> g =
      cg::tiled_partition<GROUP_SIZE>(cg::this_thread_block());

  __shared__ val_t
      penalty_here_s[QUERY_LEN]; ////RBD: have to chnge this
                                 ///!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  /* create vars for indexing */
  const idx_t block_id = blockIdx.x;
  const idx_t thread_id = cg::this_thread_block().thread_rank();
  // const idx_t base = 0; // block_id * QUERY_LEN;

  /* initialize penalties */
  val_t penalty_left = FLOAT2HALF(INFINITY);
  val_t penalty_diag = FLOAT2HALF(INFINITY);
  val_t penalty_here[SEGMENT_SIZE] = {FLOAT2HALF(INFINITY)};
  val_t penalty_temp[2];

  /* each thread computes SEGMENT_SIZE adjacent cells, get corresponding sig
   * values */
  val_t ref_coeff1[SEGMENT_SIZE], ref_coeff2[SEGMENT_SIZE];

  /* load next WARP_SIZE query values from memory into new_query_val buffer */
  val_t query_val = FLOAT2HALF(INFINITY);
  val_t new_query_val = query[block_id * QUERY_LEN + thread_id];

  /* initialize first thread's chunk */
  if (thread_id == 0) {
    query_val = new_query_val;
    penalty_diag = FLOAT2HALF(0);
  }
  new_query_val = __shfl_down_sync(ALL, new_query_val, 1);
  // for (idxt ref_batch = 0; ref_batch < REF_LEN / (REF_TILE_SIZE);
  //      ref_batch++) {
  for (idxt i = 0; i < SEGMENT_SIZE; i++) {
    // ref_coeff1[i] = ref[SEGMENT_SIZE * thread_id + i];

    ref_coeff1[i] = ref[thread_id + i * WARP_SIZE].coeff1;
    ref_coeff2[i] = ref[thread_id + i * WARP_SIZE].coeff2;
    printf("ref_coeff1[%0d]=%0f\n", i, HALF2FLOAT(ref_coeff1[i]));
  }

  /* calculate full matrix in wavefront parallel manner, multiple cells per
   * thread */
  for (idxt wave = 1; wave <= NUM_WAVES; wave++) {

    /* calculate SEGMENT_SIZE cells */
    penalty_temp[0] = penalty_here[0];
    penalty_here[0] =
        COST_FUNCTION(query_val, ref_coeff1[0], ref_coeff2[0], penalty_left,
                      penalty_here[0], penalty_diag);

    for (int i = 1; i < SEGMENT_SIZE - 2; i += 2) {
      penalty_temp[1] = penalty_here[i];
      penalty_here[i] =
          COST_FUNCTION(query_val, ref_coeff1[i], ref_coeff2[i],
                        penalty_here[i - 1], penalty_here[i], penalty_temp[0]);

      penalty_temp[0] = penalty_here[i + 1];
      penalty_here[i + 1] =
          COST_FUNCTION(query_val, ref_coeff1[i + 1], ref_coeff2[i + 1],
                        penalty_here[i], penalty_here[i + 1], penalty_temp[1]);
    }
#ifndef NV_DEBUG
    penalty_here[SEGMENT_SIZE - 1] = COST_FUNCTION(
        query_val, ref_coeff1[SEGMENT_SIZE - 1], ref_coeff2[SEGMENT_SIZE - 1],
        penalty_here[SEGMENT_SIZE - 2], penalty_here[SEGMENT_SIZE - 1],
        penalty_temp[0]);

#else
    penalty_here[SEGMENT_SIZE - 1] = COST_FUNCTION(
        query_val, ref_coeff1[SEGMENT_SIZE - 1], ref_coeff2[SEGMENT_SIZE - 1],
        FLOAT2HALF(INFINITY), penalty_here[SEGMENT_SIZE - 1], penalty_temp[0]);
#endif

    /* new_query_val buffer is empty, reload */
    if ((wave & (WARP_SIZE - 1)) == 0) {
      new_query_val = query[block_id * QUERY_LEN + wave + thread_id];
    }

    /* pass next query_value to each thread */
    query_val = __shfl_up_sync(ALL, query_val, 1);
    if (thread_id == 0) {
      query_val = new_query_val;
    }
    new_query_val = __shfl_down_sync(ALL, new_query_val, 1);

    /* transfer border cell info */
    penalty_diag = penalty_left;
    penalty_left = __shfl_up_sync(ALL, penalty_here[SEGMENT_SIZE - 1], 1);
    if (thread_id == 0) {
      penalty_left = FLOAT2HALF(INFINITY);
    }
    if ((wave >= WARP_SIZE) && (thread_id == RESULT_THREAD_ID)) {
      penalty_here_s[(wave - WARP_SIZE)] = penalty_here[RESULT_REG];
    }
  }

  printf("final_score=%0f\n", HALF2FLOAT(penalty_here[RESULT_REG]));
  /* return result */
  if ((thread_id == RESULT_THREAD_ID) && (REF_BATCH == 1)) {
    // printf("@@@result_threadId=%0d\n",RESULT_THREAD_ID);

    dist[block_id] =
        penalty_here[RESULT_REG] > thresh ? FLOAT2HALF(0) : FLOAT2HALF(1);
    return;
  }

  /*------------------------------for all ref batches > 0
   * ---------------------------------- */
  for (idxt ref_batch = 1; ref_batch < REF_BATCH; ref_batch++) {
    /* initialize penalties */
    penalty_left = FLOAT2HALF(INFINITY);
    penalty_diag = FLOAT2HALF(INFINITY);
    for (auto i = 0; i < SEGMENT_SIZE; i++)
      penalty_here[i] = FLOAT2HALF(INFINITY);
    for (auto i = 0; i < 2; i++)
      penalty_temp[i] = FLOAT2HALF(INFINITY);

    /* load next WARP_SIZE query values from memory into new_query_val buffer */
    query_val = FLOAT2HALF(INFINITY);
    new_query_val = query[block_id * QUERY_LEN +
                          QUERY_LEN * (ref_batch) / REF_BATCH + thread_id];

    /* initialize first thread's chunk */
    if (thread_id == 0) {
      query_val = new_query_val;
      penalty_diag = FLOAT2HALF(0);
      penalty_left = penalty_here_s[0];
    }
    new_query_val = __shfl_down_sync(ALL, new_query_val, 1);

    for (idxt i = 0; i < SEGMENT_SIZE; i++) {
      ref_coeff1[i] =
          ref[ref_batch * (REF_TILE_SIZE) + SEGMENT_SIZE * thread_id + i]
              .coeff1;
      ref_coeff2[i] =
          ref[ref_batch * (REF_TILE_SIZE) + SEGMENT_SIZE * thread_id + i]
              .coeff2;
    }
    /* calculate full matrix in wavefront parallel manner, multiple cells per
     * thread */
    for (idxt wave = 1; wave <= NUM_WAVES; wave++) {

      /* calculate SEGMENT_SIZE cells */
      penalty_temp[0] = penalty_here[0];
      penalty_here[0] =
          COST_FUNCTION(query_val, ref_coeff1[0], ref_coeff2[0], penalty_left,
                        penalty_here[0], penalty_diag);

      for (int i = 1; i < SEGMENT_SIZE - 2; i += 2) {
        penalty_temp[1] = penalty_here[i];
        penalty_here[i] = COST_FUNCTION(query_val, ref_coeff1[i], ref_coeff2[i],
                                        penalty_here[i - 1], penalty_here[i],
                                        penalty_temp[0]);

        penalty_temp[0] = penalty_here[i + 1];
        penalty_here[i + 1] = COST_FUNCTION(
            query_val, ref_coeff1[i + 1], ref_coeff2[i + 1], penalty_here[i],
            penalty_here[i + 1], penalty_temp[1]);
      }
#ifndef NV_DEBUG
      penalty_here[SEGMENT_SIZE - 1] = COST_FUNCTION(
          query_val, ref_coeff1[SEGMENT_SIZE - 1], ref_coeff2[SEGMENT_SIZE - 1],
          penalty_here[SEGMENT_SIZE - 2], penalty_here[SEGMENT_SIZE - 1],
          penalty_temp[0]);

#else
      penalty_here[SEGMENT_SIZE - 1] =
          COST_FUNCTION(query_val, ref_coeff1[SEGMENT_SIZE - 1],
                        ref_coeff2[SEGMENT_SIZE - 1], FLOAT2HALF(INFINITY),
                        penalty_here[SEGMENT_SIZE - 1], penalty_temp[0]);
#endif

      /* new_query_val buffer is empty, reload */
      if ((wave & (WARP_SIZE - 1)) == 0) {
        new_query_val = query[block_id * QUERY_LEN + wave + thread_id];
      }

      /* pass next query_value to each thread */
      query_val = __shfl_up_sync(ALL, query_val, 1);
      if (thread_id == 0) {
        query_val = new_query_val;
      }
      new_query_val = __shfl_down_sync(ALL, new_query_val, 1);

      /* transfer border cell info */
      penalty_diag = penalty_left;
      penalty_left = __shfl_up_sync(ALL, penalty_here[SEGMENT_SIZE - 1], 1);
      if (thread_id == 0) {
        penalty_left = penalty_here_s[wave];
      }
      if ((wave >= WARP_SIZE) && (thread_id == RESULT_THREAD_ID)) {
        penalty_here_s[(wave - WARP_SIZE)] = penalty_here[RESULT_REG];
      }
    }
    /* return result */
    if ((thread_id == RESULT_THREAD_ID) && (ref_batch == (REF_BATCH - 1))) {
      // printf("@@@result_threadId=%0d\n",RESULT_THREAD_ID);

      dist[block_id] =
          penalty_here[RESULT_REG] > thresh ? FLOAT2HALF(0) : FLOAT2HALF(1);

      return;
    }
  }
}

#endif //} //DTW

#endif //}