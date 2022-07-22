#ifndef FULLDTW
#define FULLDTW

#include "common.hpp"
#include "datatypes.hpp"
#include <cooperative_groups.h>

#ifdef FP16 // FP16 definitions

#include <cuda_fp16.h>

#endif

namespace cg = cooperative_groups;
#define ALL 0xFFFFFFFF

#define COST_FUNCTION(q, r, l, t, d)                                           \
  FMA(FMA(r, FLOAT2HALF(-1), q), FMA(r, FLOAT2HALF(-1), q),                    \
      FIND_MIN(l, FIND_MIN(t, d)))

#ifndef SDTW

template <typename index_t, typename val_t>
__global__ void DTW(reference_coefficients *ref, val_t *query, val_t *dist,
                    index_t num_entries, val_t thresh) {

  // cooperative threading
  cg::thread_block_tile<GROUP_SIZE> g =
      cg::tiled_partition<GROUP_SIZE>(cg::this_thread_block());

  __shared__ val_t
      penalty_here_s[QUERY_LEN]; ////RBD: have to chnge this
                                 ///!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  /* create vars for indexing */
  const index_t block_id = blockIdx.x;
  const index_t thread_id = cg::this_thread_block().thread_rank();
  // const index_t base = 0; // block_id * QUERY_LEN;

  /* initialize penalties */
  val_t penalty_left = FLOAT2HALF(INFINITY);
  val_t penalty_diag = FLOAT2HALF(INFINITY);
  val_t penalty_here[SEGMENT_SIZE] = {FLOAT2HALF(INFINITY)};
  val_t penalty_temp[2];

  /* each thread computes CELLS_PER_THREAD adjacent cells, get corresponding sig
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
  // for (idxt ref_batch = 0; ref_batch < REF_LEN / (SEGMENT_SIZE * WARP_SIZE);
  //      ref_batch++) {
  for (idxt i = 0; i < SEGMENT_SIZE; i++) {
    // ref_coeff1[i] = ref[CELLS_PER_THREAD * thread_id + i];

    ref_coeff1[i] = ref[thread_id + i * WARP_SIZE].coeff1;
    ref_coeff2[i] = ref[thread_id + i * WARP_SIZE].coeff2;
    printf("ref_coeff1[%0d]=%0f\n", i, HALF2FLOAT(ref_coeff1[i]));
  }

  /* calculate full matrix in wavefront parallel manner, multiple cells per
   * thread */
  for (idxt wave = 1; wave <= NUM_WAVES; wave++) {

    /* calculate CELLS_PER_THREAD cells */
    penalty_temp[0] = penalty_here[0];
    penalty_here[0] = COST_FUNCTION(query_val, ref_coeff1[0], penalty_left,
                                    penalty_here[0], penalty_diag);

    for (int i = 1; i < SEGMENT_SIZE - 2; i += 2) {
      penalty_temp[1] = penalty_here[i];
      penalty_here[i] =
          COST_FUNCTION(query_val, ref_coeff1[i], penalty_here[i - 1],
                        penalty_here[i], penalty_temp[0]);

      penalty_temp[0] = penalty_here[i + 1];
      penalty_here[i + 1] =
          COST_FUNCTION(query_val, ref_coeff1[i + 1], penalty_here[i - 1],
                        penalty_here[i + 1], penalty_temp[1]);
    }
#ifndef NV_DEBUG
    penalty_here[SEGMENT_SIZE - 1] = COST_FUNCTION(
        query_val, ref_coeff1[SEGMENT_SIZE - 1], penalty_here[SEGMENT_SIZE - 2],
        penalty_here[SEGMENT_SIZE - 1], penalty_temp[0]);

#else
    penalty_here[SEGMENT_SIZE - 1] = COST_FUNCTION(
        query_val, ref_coeff1[SEGMENT_SIZE - 1], FLOAT2HALF(INFINITY),
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
      penalty_left = FLOAT2HALF(INFINITY);
    }
    if ((wave >= WARP_SIZE) && (thread_id == RESULT_THREAD_ID)) {
      penalty_here_s[(wave - WARP_SIZE)] = penalty_here[RESULT_REG];
    }
  }

  printf("final_score=%0f\n", HALF2FLOAT(penalty_here[RESULT_REG]));
  /* return result */
  if ((thread_id == RESULT_THREAD_ID) && (REF_BATCH == 1)) {
    // printf("@@@result_threadId=%0ld\n",RESULT_THREAD_ID);

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
      ref_coeff1[i] = ref[ref_batch * (SEGMENT_SIZE * WARP_SIZE) +
                          CELLS_PER_THREAD * thread_id + i]
                          .coeff1;
      ref_coeff2[i] = ref[ref_batch * (SEGMENT_SIZE * WARP_SIZE) +
                          CELLS_PER_THREAD * thread_id + i]
                          .coeff2;
    }
    /* calculate full matrix in wavefront parallel manner, multiple cells per
     * thread */
    for (idxt wave = 1; wave <= NUM_WAVES; wave++) {

      /* calculate CELLS_PER_THREAD cells */
      penalty_temp[0] = penalty_here[0];
      penalty_here[0] = COST_FUNCTION(query_val, ref_coeff1[0], penalty_left,
                                      penalty_here[0], penalty_diag);

      for (int i = 1; i < SEGMENT_SIZE - 2; i += 2) {
        penalty_temp[1] = penalty_here[i];
        penalty_here[i] =
            COST_FUNCTION(query_val, ref_coeff1[i], penalty_here[i - 1],
                          penalty_here[i], penalty_temp[0]);

        penalty_temp[0] = penalty_here[i + 1];
        penalty_here[i + 1] =
            COST_FUNCTION(query_val, ref_coeff1[i + 1], penalty_here[i - 1],
                          penalty_here[i + 1], penalty_temp[1]);
      }
#ifndef NV_DEBUG
      penalty_here[SEGMENT_SIZE - 1] =
          COST_FUNCTION(query_val, ref_coeff1[SEGMENT_SIZE - 1],
                        penalty_here[SEGMENT_SIZE - 2],
                        penalty_here[SEGMENT_SIZE - 1], penalty_temp[0]);

#else
      penalty_here[SEGMENT_SIZE - 1] = COST_FUNCTION(
          query_val, ref_coeff1[SEGMENT_SIZE - 1], FLOAT2HALF(INFINITY),
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
      // printf("@@@result_threadId=%0ld\n",RESULT_THREAD_ID);

      dist[block_id] =
          penalty_here[RESULT_REG] > thresh ? FLOAT2HALF(0) : FLOAT2HALF(1);

      return;
    }
  }
}

/*----------------------------------subsequence
 * DTW--------------------------------*/

#else //{
template <typename index_t, typename val_t>
__global__ void DTW(reference_coefficients *ref, val_t *query, val_t *dist,
                    index_t num_entries, val_t thresh) {

  // cooperative threading
  cg::thread_block_tile<GROUP_SIZE> g =
      cg::tiled_partition<GROUP_SIZE>(cg::this_thread_block());

  __shared__ val_t
      penalty_here_s[QUERY_LEN]; ////RBD: have to chnge this
                                 ///!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  /* create vars for indexing */
  const index_t block_id = blockIdx.x;
  const index_t thread_id = cg::this_thread_block().thread_rank();
  // const index_t base = 0; // block_id * QUERY_LEN;

  /* initialize penalties */
  val_t penalty_left = FLOAT2HALF(INFINITY);
  val_t penalty_diag = FLOAT2HALF(INFINITY);
  val_t penalty_here[SEGMENT_SIZE] = {FLOAT2HALF(0)};
  val_t penalty_temp[2];
  val_t min_segment = FLOAT2HALF(INFINITY); // finds min of segment for sDTW

  /* each thread computes CELLS_PER_THREAD adjacent cells, get corresponding sig
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
  // for (idxt ref_batch = 0; ref_batch < REF_LEN / (SEGMENT_SIZE * WARP_SIZE);
  //      ref_batch++) {
  for (idxt i = 0; i < SEGMENT_SIZE; i++) {
    ref_coeff1[i] = ref[thread_id + i * WARP_SIZE].coeff1;
    ref_coeff2[i] = ref[thread_id + i * WARP_SIZE].coeff2;
  }
  /* calculate full matrix in wavefront parallel manner, multiple cells per
   * thread */
  for (idxt wave = 1; wave <= NUM_WAVES; wave++) {

    /* calculate CELLS_PER_THREAD cells */
    penalty_temp[0] = penalty_here[0];
    penalty_here[0] = COST_FUNCTION(query_val, ref_coeff1[0], penalty_left,
                                    penalty_here[0], penalty_diag);

    for (int i = 1; i < SEGMENT_SIZE - 2; i += 2) {
      penalty_temp[1] = penalty_here[i];
      penalty_here[i] =
          COST_FUNCTION(query_val, ref_coeff1[i], penalty_here[i - 1],
                        penalty_here[i], penalty_temp[0]);

      penalty_temp[0] = penalty_here[i + 1];
      penalty_here[i + 1] =
          COST_FUNCTION(query_val, ref_coeff1[i + 1], penalty_here[i - 1],
                        penalty_here[i + 1], penalty_temp[1]);
    }
#ifndef NV_DEBUG
    penalty_here[SEGMENT_SIZE - 1] = COST_FUNCTION(
        query_val, ref_coeff1[SEGMENT_SIZE - 1], penalty_here[SEGMENT_SIZE - 2],
        penalty_here[SEGMENT_SIZE - 1], penalty_temp[0]);
#else
    penalty_here[SEGMENT_SIZE - 1] = COST_FUNCTION(
        query_val, ref_coeff1[SEGMENT_SIZE - 1], FLOAT2HALF(INFINITY),
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
      penalty_left = FLOAT2HALF(INFINITY);
    }
    if ((wave >= WARP_SIZE) && (thread_id == RESULT_THREAD_ID)) {
      penalty_here_s[(wave - WARP_SIZE)] = penalty_here[RESULT_REG];
    }
    // Find min of segment and then shuffle up for sDTW
    if (wave >= QUERY_LEN) {
      for (idxt i = 0; i < SEGMENT_SIZE; i++) {
        min_segment = FIND_MIN(min_segment, penalty_here[i]);
      }
      min_segment = __shfl_up_sync(ALL, min_segment, 1);
    }
  }

  /* return result */

  if ((thread_id == RESULT_THREAD_ID) && (REF_BATCH == 1)) {
    // printf("@@@result_threadId=%0ld\n",RESULT_THREAD_ID);

    // dist[block_id] =
    //     penalty_here[RESULT_REG] > thresh ? FLOAT2HALF(0) : FLOAT2HALF(1);
    dist[block_id] = HALF2FLOAT(min_segment);
    printf("score=%0f,min_segment=%0f\n", penalty_here[RESULT_REG],
           min_segment);
    return;
  }

  /*------------------------------for all ref batches > 0
   * ---------------------------------- */
  for (idxt ref_batch = 1; ref_batch < REF_BATCH; ref_batch++) {
    /* initialize penalties */
    penalty_left = FLOAT2HALF(INFINITY);
    penalty_diag = FLOAT2HALF(INFINITY);
    for (auto i = 0; i < SEGMENT_SIZE; i++)
      penalty_here[i] = FLOAT2HALF(0);
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
      ref_coeff1[i] = ref[ref_batch * (SEGMENT_SIZE * WARP_SIZE) +
                          CELLS_PER_THREAD * thread_id + i]
                          .coeff1;
      ref_coeff2[i] = ref[ref_batch * (SEGMENT_SIZE * WARP_SIZE) +
                          CELLS_PER_THREAD * thread_id + i]
                          .coeff2;
    }
    /* calculate full matrix in wavefront parallel manner, multiple cells per
     * thread */
    for (idxt wave = 1; wave <= NUM_WAVES; wave++) {

      /* calculate CELLS_PER_THREAD cells */
      penalty_temp[0] = penalty_here[0];
      penalty_here[0] = COST_FUNCTION(query_val, ref_coeff1[0], penalty_left,
                                      penalty_here[0], penalty_diag);

      for (int i = 1; i < SEGMENT_SIZE - 2; i += 2) {
        penalty_temp[1] = penalty_here[i];
        penalty_here[i] =
            COST_FUNCTION(query_val, ref_coeff1[i], penalty_here[i - 1],
                          penalty_here[i], penalty_temp[0]);

        penalty_temp[0] = penalty_here[i + 1];
        penalty_here[i + 1] =
            COST_FUNCTION(query_val, ref_coeff1[i + 1], penalty_here[i - 1],
                          penalty_here[i + 1], penalty_temp[1]);
      }
#ifndef NV_DEBUG
      penalty_here[SEGMENT_SIZE - 1] =
          COST_FUNCTION(query_val, ref_coeff1[SEGMENT_SIZE - 1],
                        penalty_here[SEGMENT_SIZE - 2],
                        penalty_here[SEGMENT_SIZE - 1], penalty_temp[0]);
#else
      penalty_here[SEGMENT_SIZE - 1] = COST_FUNCTION(
          query_val, ref_coeff1[SEGMENT_SIZE - 1], FLOAT2HALF(INFINITY),
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
      // Find min of segment and then shuffle up for sDTW
      if (wave >= QUERY_LEN) {
        for (idxt i = 0; i < SEGMENT_SIZE; i++) {
          min_segment = FIND_MIN(min_segment, penalty_here[i]);
        }
        min_segment = __shfl_up_sync(ALL, min_segment, 1);
      }
    }
    /* return result */
    if ((thread_id == RESULT_THREAD_ID) && (ref_batch == (REF_BATCH - 1))) {
      // printf("@@@result_threadId=%0ld\n",RESULT_THREAD_ID);

      // dist[block_id] =
      //     penalty_here[RESULT_REG] > thresh ? FLOAT2HALF(0) : FLOAT2HALF(1);
      dist[block_id] = HALF2FLOAT(min_segment);

      return;
    }
  }
}
#endif //} //SDTW

#endif //}