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
// #define COST_FUNCTION(q, r1, r2, l, t, d)                                      \
//   FMA(FMA(FMA(r1, FLOAT2HALF2(-1), q), FMA(r1, FLOAT2HALF2(-1), q), 0), r2,      \
//       FIND_MIN(l, FIND_MIN(t, d)))

#ifndef NO_REF_DEL
#define COST_FUNCTION(q, r1, l, t, d)                                          \
  FMA(FMA(SUB(r1, q), SUB(r1, q), FLOAT2HALF2(0.0f)), FLOAT2HALF2(1.0f),       \
      FIND_MIN(l, FIND_MIN(t, d)))                                             \
  // FMA(FMA(FMA(r1, FLOAT2HALF2(-1.0f), q), FMA(r1, FLOAT2HALF2(-1.0f), q),      \
  //         FLOAT2HALF2(0.0f)),                                                  \
  //     FLOAT2HALF2(1.0f), FIND_MIN(l, FIND_MIN(t, d)))

#else // assuming there are no reference
// #define COST_FUNCTION(q, r1, l, t, d)                                          \
  // FMA(FMA(FMA(r1, FLOAT2HALF2(-1.0f), q), FMA(r1, FLOAT2HALF2(-1.0f), q),      \
  //         FLOAT2HALF2(0.0f)),                                                  \
  //     FLOAT2HALF2(1.0f), FIND_MIN(t, d))

#define COST_FUNCTION(q, r1, l, t, d)                                          \
  FMA(FMA(SUB(r1, q), SUB(r1, q), FLOAT2HALF2(0.0f)), FLOAT2HALF2(1.0f),       \
      FIND_MIN(t, d))
#endif

// computes segments of the sDTW matrix
template <typename idx_t, typename val_t>
__device__ __forceinline__ void
compute_segment(idxt &wave, const idx_t &thread_id, val_t &query_val,
                val_t (&ref_coeff1)[SEGMENT_SIZE], val_t &penalty_left,
                val_t (&penalty_here)[SEGMENT_SIZE], val_t &penalty_diag,
                val_t (&penalty_temp)[2], idxt query_batch) {
  /* calculate SEGMENT_SIZE cells */
  penalty_temp[0] = penalty_here[0];

  if ((thread_id != (wave - 1)) || (query_batch)) {
#ifdef NV_DEBUG
#ifndef FP16
    printf(
        "wave= %0d, tid=%0d, query= %0f, ref1= %0f, penalty_here[%0d]= %0f,   \
      penalty_left = % 0f, \
      penalty_diag = % 0f\n",
        wave, thread_id, HALF2FLOAT(query_val), HALF2FLOAT(ref_coeff1[REG_ID]),
        REG_ID, HALF2FLOAT(penalty_here[REG_ID]), HALF2FLOAT(penalty_left),
        HALF2FLOAT(penalty_diag));
#else
    printf(
        "wave= %0d, tid=%0d, query= %0f, ref1= %0f, penalty_here[%0d]= %0f,   \
        penalty_left = % 0f, \
        penalty_diag = % 0f\n",
        wave, thread_id, HALF2FLOAT(query_val.x),
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
        wave, thread_id, HALF2FLOAT(query_val), HALF2FLOAT(ref_coeff1[REG_ID]),
        REG_ID, HALF2FLOAT(penalty_here[REG_ID]), HALF2FLOAT(penalty_left),
        HALF2FLOAT(penalty_diag));
#else
    printf(
        "wave= %0d, tid=%0d, query= %0f, ref1= %0f, penalty_here[%0d]= %0f,   \
        penalty_left = % 0f, \
        penalty_diag = % 0f\n",
        wave, thread_id, HALF2FLOAT(query_val.x),
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

/*----------------------------------subsequence
 * DTW--------------------------------*/
template <typename idx_t, typename val_t>
__global__ void DTW(reference_coefficients *ref, val_t *query, val_t *dist,
                    idx_t num_entries, val_t thresh, val_t *device_last_row) {

#if REF_BATCH > 1
  __shared__ val_t penalty_last_col[PREFIX_LEN]; ////RBD: have to chnge this
#endif ///!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  /* create vars for indexing */
  const idx_t block_id = blockIdx.x;
  // const idx_t thread_id = cg::this_thread_block().thread_rank();
  const idx_t thread_id = threadIdx.x;
  // const idx_t base = 0; // block_id * PREFIX_LEN;

  /* initialize penalties */

  val_t penalty_temp[2];
  val_t min_segment = FLOAT2HALF2(INFINITY); // finds min of segment for sDTW
  val_t last_col_penalty_shuffled;           // used to store last col of matrix

  /* each thread computes SEGMENT_SIZE adjacent cells, get corresponding sig
   * values */
  val_t ref_coeff1[SEGMENT_SIZE]; // ref_coeff2[SEGMENT_SIZE];

/* load next WARP_SIZE query values from memory into new_query_val buffer */
#pragma unroll
  for (idxt query_batch = 0; query_batch < QUERY_BATCH; query_batch++) {

    val_t penalty_left = FLOAT2HALF2(INFINITY);
    val_t penalty_diag = FLOAT2HALF2(INFINITY);
    val_t penalty_here[SEGMENT_SIZE] = {FLOAT2HALF2(0)};
    val_t query_val = FLOAT2HALF2(INFINITY);
    val_t new_query_val =
        query[(block_id * QUERY_LEN) + (query_batch * PREFIX_LEN) + thread_id];
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
      // ref_coeff2[i] = ref[thread_id + i * WARP_SIZE].coeff2;
#ifdef NV_DEBUG
#ifndef FP16
      printf("tid= %0d, ref_coeff1[%0d]=%0f\n", thread_id, i,
             HALF2FLOAT(ref_coeff1[i]));
#else
      printf("tid= %0d, ref_coeff1[%0d]=%0f\n", thread_id, i,
             HALF2FLOAT(ref_coeff1[i].x));
#endif
#endif
#if QUERY_BATCH > 1
      if (query_batch > 0)
        penalty_here[i] =
            device_last_row[block_id * REF_LEN + thread_id + i * WARP_SIZE];
#endif
    }

    /* calculate full matrix in wavefront parallel manner, multiple cells per
     * thread */
    for (idxt wave = 1; wave <= NUM_WAVES; wave++) {
      if (query_batch == (QUERY_BATCH - 1))
        min_segment = __shfl_up_sync((ALL), min_segment, 1);

      if (((wave - PREFIX_LEN) <= thread_id) &&
          (thread_id <=
           (wave - 1))) // HS: block cells that have completed from further
        compute_segment<idx_t, val_t>(wave, thread_id, query_val, ref_coeff1,
                                      penalty_left, penalty_here, penalty_diag,
                                      penalty_temp, query_batch);

      /* new_query_val buffer is empty, reload */
      if ((wave & (WARP_SIZE_MINUS_ONE)) == 0) {
        // if (block_id * PREFIX_LEN + wave + thread_id > 32075775)

        new_query_val = query[(block_id * QUERY_LEN) +
                              (query_batch * PREFIX_LEN) + wave + thread_id];
      }

      /* pass next query_value to each thread */
      query_val = __shfl_up_sync(ALL, query_val, 1);

      /* transfer border cell info */

      penalty_diag = penalty_left;

      penalty_left = __shfl_up_sync(ALL, penalty_here[SEGMENT_SIZE - 1], 1);
      if (thread_id == 0) {
        query_val = new_query_val;
        penalty_left = FLOAT2HALF2(INFINITY);
      }
      new_query_val = __shfl_down_sync(ALL, new_query_val, 1);

#if REF_BATCH > 1
      // if ((wave >= WARP_SIZE) && (thread_id == WARP_SIZE_MINUS_ONE)) {
      //   penalty_last_col[(wave - WARP_SIZE)] = penalty_here[RESULT_REG];
      // }
      last_col_penalty_shuffled =
          __shfl_down_sync(ALL, last_col_penalty_shuffled, 1);
      if (thread_id == WARP_SIZE_MINUS_ONE)
        last_col_penalty_shuffled = penalty_here[RESULT_REG];
      if ((wave >= TWICE_WARP_SIZE_MINUS_ONE) &&
          ((wave & WARP_SIZE_MINUS_ONE) == WARP_SIZE_MINUS_ONE)) { // HS
        penalty_last_col[(wave - TWICE_WARP_SIZE_MINUS_ONE) + thread_id] =
            last_col_penalty_shuffled;
#ifdef NV_DEBUG
        printf(
            "smem write,querybatch=%0d, i=%0d,value=%0f\n ", query_batch,
            (wave - TWICE_WARP_SIZE_MINUS_ONE) + thread_id,
            penalty_last_col[(wave - TWICE_WARP_SIZE_MINUS_ONE) + thread_id]);
#endif
      } else if ((wave >= NUM_WAVES_BY_WARP_SIZE) &&
                 (thread_id == WARP_SIZE_MINUS_ONE)) {
        penalty_last_col[wave - WARP_SIZE] = penalty_here[RESULT_REG];
#ifdef NV_DEBUG
        printf("smem write,querybatch=%0d, i=%0d,value=%0f\n ", query_batch,
               wave - WARP_SIZE,
               penalty_last_col[(wave - TWICE_WARP_SIZE_MINUS_ONE)]);
#endif
      }

#endif
      // Find min of segment and then shuffle up for sDTW
      if ((wave >= PREFIX_LEN) && (query_batch == (QUERY_BATCH - 1))) {
        for (idxt i = 0; i < SEGMENT_SIZE; i++) {
          min_segment = FIND_MIN(min_segment, penalty_here[i]);
        }
#ifdef NV_DEBUG
        if (thread_id == (wave - PREFIX_LEN))
          printf(
              "minsegment=%0f,tid=%0d,wave=%0d, refbatch=%0d,querybatch=%0d\n",
              min_segment, thread_id, wave, 0, query_batch);
#endif
      }
    }

// write last row to smem
#if QUERY_BATCH > 1
    for (idxt i = 0; i < SEGMENT_SIZE; i++) {
      device_last_row[block_id * REF_LEN + thread_id + i * WARP_SIZE] =
          penalty_here[i];
#ifdef NV_DEBUG
      printf("last row idx=%0d, val=%0f\n", thread_id + i * WARP_SIZE,
             penalty_here[i]);
#endif
    }
#endif

    /*------------------------------for all ref batches > 0
     * ---------------------------------- */
#if REF_BATCH > 1
    for (idxt ref_batch = 1; ref_batch < REF_BATCH_MINUS_ONE; ref_batch++) {

#ifdef NV_DEBUG
      printf("refbatch=%0d\n", ref_batch);
#endif
      if (query_batch == (QUERY_BATCH - 1))
        min_segment = __shfl_down_sync((ALL), min_segment, 31);

      /* initialize penalties */
      // val_t penalty_left = FLOAT2HALF2(INFINITY);
      penalty_diag = FLOAT2HALF2(INFINITY);
      penalty_left = FLOAT2HALF2(INFINITY);

      if (query_batch == 0) {
        for (auto i = 0; i < SEGMENT_SIZE; i++)
          penalty_here[i] = FLOAT2HALF2(0);

      } else {
        for (auto i = 0; i < SEGMENT_SIZE; i++)
          penalty_here[i] =
              device_last_row[block_id * REF_LEN + ref_batch * REF_TILE_SIZE +
                              thread_id + i * WARP_SIZE];
      }

      // for (auto i = 0; i < 2; i++)
      //   penalty_temp[i] = FLOAT2HALF2(INFINITY);

      /* load next WARP_SIZE query values from memory into new_query_val buffer
       */
      query_val = FLOAT2HALF2(INFINITY);
      new_query_val = query[(block_id * QUERY_LEN) +
                            (query_batch * PREFIX_LEN) + thread_id];

      /* initialize first thread's chunk */
      if (thread_id == 0) {
        query_val = new_query_val;

        penalty_left = penalty_last_col[0];
      }
      new_query_val = __shfl_down_sync(ALL, new_query_val, 1);

      for (idxt i = 0; i < SEGMENT_SIZE; i++) {
        ref_coeff1[i] =
            ref[ref_batch * (REF_TILE_SIZE) + thread_id + i * WARP_SIZE].coeff1;
        // ref_coeff2[i] =
        //   ref[ref_batch * (REF_TILE_SIZE) + thread_id + i *
        //   WARP_SIZE].coeff2;
      }
      /* calculate full matrix in wavefront parallel manner, multiple cells per
       * thread */

      for (idxt wave = 1; wave <= NUM_WAVES; wave++) {

        if (((wave - PREFIX_LEN) <= thread_id) &&
            (thread_id <=
             (wave - 1))) // HS: block cells that have completed from further
          compute_segment<idx_t, val_t>(
              wave, thread_id, query_val, ref_coeff1, penalty_left,
              penalty_here, penalty_diag, penalty_temp, query_batch);

        /* new_query_val buffer is empty, reload */
        if ((wave & (WARP_SIZE_MINUS_ONE)) == 0) {
          new_query_val = query[(block_id * QUERY_LEN) +
                                (query_batch * PREFIX_LEN) + wave + thread_id];
        }

        /* pass next query_value to each thread */
        query_val = __shfl_up_sync(ALL, query_val, 1);
        if (thread_id == 0) {
          query_val = new_query_val;
        }

        last_col_penalty_shuffled =
            __shfl_down_sync(ALL, last_col_penalty_shuffled, 1);
        if (thread_id == WARP_SIZE_MINUS_ONE)
          last_col_penalty_shuffled = penalty_here[RESULT_REG];
        if ((wave >= TWICE_WARP_SIZE_MINUS_ONE) &&
            ((wave & WARP_SIZE_MINUS_ONE) == WARP_SIZE_MINUS_ONE)) // HS
          penalty_last_col[(wave - TWICE_WARP_SIZE_MINUS_ONE) + thread_id] =
              last_col_penalty_shuffled;
        else if ((wave >= NUM_WAVES_BY_WARP_SIZE) &&
                 (thread_id == WARP_SIZE_MINUS_ONE))
          penalty_last_col[(wave - TWICE_WARP_SIZE_MINUS_ONE)] =
              penalty_here[RESULT_REG];
        new_query_val = __shfl_down_sync(ALL, new_query_val, 1);

        /* transfer border cell info */
        penalty_diag = penalty_left;
        penalty_left = __shfl_up_sync(ALL, penalty_here[SEGMENT_SIZE - 1], 1);

        if (thread_id == 0) {

          penalty_left = penalty_last_col[wave];
        }

        // Find min of segment and then shuffle up for sDTW
        if ((wave >= PREFIX_LEN) && (query_batch == (QUERY_BATCH - 1))) {
          for (idxt i = 0; i < SEGMENT_SIZE; i++) {
            min_segment = FIND_MIN(min_segment, penalty_here[i]);
          }
#ifdef NV_DEBUG
          if (thread_id == (wave - PREFIX_LEN))
            printf("minsegment=%0f,tid=%0d,wave=%0d, "
                   "refbatch=%0d,querybatch=%0d\n",
                   min_segment, thread_id, wave, ref_batch, query_batch);
#endif
          if (wave != (NUM_WAVES))
            min_segment = __shfl_up_sync((ALL), min_segment, 1);
        }
      }

// write last row to smem
#if QUERY_BATCH > 1
      for (idxt i = 0; i < SEGMENT_SIZE; i++) {
        device_last_row[block_id * REF_LEN + ref_batch * REF_TILE_SIZE +
                        thread_id + i * WARP_SIZE] = penalty_here[i];
#ifdef NV_DEBUG
        printf("penalty last row write, last ref batc i=%0d,tid=%0d, val=%0f\n",
               block_id * REF_LEN + REF_LEN - REF_TILE_SIZE + thread_id +
                   i * WARP_SIZE,
               thread_id, penalty_here[i]);
#endif
      }
#endif
    }

//----------------------------last
// sub-matrix calculation or
// ref_batch=REF_BATCH-1------------------------------------------------//
#ifdef NV_DEBUG
    printf("refbatch=%0d\n", REF_BATCH_MINUS_ONE);
#endif
    if (query_batch == (QUERY_BATCH - 1))
      min_segment = __shfl_down_sync((ALL), min_segment, 31);

    /* initialize penalties */
    // val_t penalty_left = FLOAT2HALF2(INFINITY);
    penalty_diag = FLOAT2HALF2(INFINITY);
    penalty_left = FLOAT2HALF2(INFINITY);

    if (query_batch == 0) {
      for (auto i = 0; i < SEGMENT_SIZE; i++)
        penalty_here[i] = FLOAT2HALF2(0);
    } else {
      for (auto i = 0; i < SEGMENT_SIZE; i++) {
        penalty_here[i] =
            device_last_row[block_id * REF_LEN + REF_LEN - REF_TILE_SIZE +
                            thread_id + i * WARP_SIZE];
#ifdef NV_DEBUG
        printf("gmem read penalty last row last ref batch i=%0d,tid=%0d, "
               "val=%0f\n",
               block_id * REF_LEN + REF_LEN - REF_TILE_SIZE + thread_id +
                   i * WARP_SIZE,
               thread_id, penalty_here[i]);
#endif
      }
    }

    // for (auto i = 0; i < 2; i++)
    //   penalty_temp[i] = FLOAT2HALF2(INFINITY);

    /* load next WARP_SIZE query values from memory into new_query_val buffer */
    query_val = FLOAT2HALF2(INFINITY);
    new_query_val =
        query[(block_id * QUERY_LEN) + (query_batch * PREFIX_LEN) + thread_id];

    /* initialize first thread's chunk */
    if (thread_id == 0) {
      query_val = new_query_val;

      penalty_left = penalty_last_col[0];
#ifdef NV_DEBUG
      printf("reading penalty_last_col[0]=%0f\n", penalty_last_col[0]);
#endif
    }
    new_query_val = __shfl_down_sync(ALL, new_query_val, 1);

    for (idxt i = 0; i < SEGMENT_SIZE; i++) {
      ref_coeff1[i] =
          ref[REF_BATCH_MINUS_ONE * (REF_TILE_SIZE) + thread_id + i * WARP_SIZE]
              .coeff1;
      // ref_coeff2[i] =
      //   ref[ref_batch * (REF_TILE_SIZE) + thread_id + i * WARP_SIZE].coeff2;
    }
    /* calculate full matrix in wavefront parallel manner, multiple cells per
     * thread */

    for (idxt wave = 1; wave <= NUM_WAVES; wave++) {

      if (((wave - PREFIX_LEN) <= thread_id) &&
          (thread_id <=
           (wave - 1))) // HS: block cells that have completed from further
        compute_segment<idx_t, val_t>(wave, thread_id, query_val, ref_coeff1,
                                      penalty_left, penalty_here, penalty_diag,
                                      penalty_temp, query_batch);

      /* new_query_val buffer is empty, reload */
      if ((wave & (WARP_SIZE_MINUS_ONE)) == 0) {
        new_query_val = query[(block_id * QUERY_LEN) +
                              (query_batch * PREFIX_LEN) + wave + thread_id];
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

        penalty_left = penalty_last_col[wave];
#ifdef NV_DEBUG
        printf("reading penalty_last_col[%0d]=%0f\n", wave,
               penalty_last_col[wave]);
#endif
      }

      // Find min of segment and then shuffle up for sDTW
      if ((wave >= PREFIX_LEN) && (query_batch == (QUERY_BATCH - 1))) {
        for (idxt i = 0; i < SEGMENT_SIZE; i++) {
          min_segment = FIND_MIN(min_segment, penalty_here[i]);
        }
#ifdef NV_DEBUG
        if (thread_id == (wave - PREFIX_LEN))
          printf(
              "minsegment=%0f,tid=%0d,wave=%0d, refbatch=%0d,querybatch=%0d\n",
              min_segment, thread_id, wave, REF_BATCH_MINUS_ONE, query_batch);
#endif
        if (wave != (NUM_WAVES))
          min_segment = __shfl_up_sync((ALL), min_segment, 1);
      }
    }
    // write last row to smem
#if QUERY_BATCH > 1
    for (idxt i = 0; i < SEGMENT_SIZE; i++) {
      device_last_row[block_id * REF_LEN + REF_LEN - REF_TILE_SIZE + thread_id +
                      i * WARP_SIZE] = penalty_here[i];
#ifdef NV_DEBUG
      printf("gmem write, last row idx=%0d, val=%0f\n",
             block_id * REF_LEN + REF_LEN - REF_TILE_SIZE + thread_id +
                 i * WARP_SIZE,
             penalty_here[i]);
#endif
    }
#endif

#endif

  } // query_batch loop
  //-------------------------------------------------------------------------------------------------------------------//
  /* return result */
  if (thread_id == WARP_SIZE_MINUS_ONE) {

    dist[block_id] = min_segment;

#ifdef NV_DEBUG
    printf("min_segment=%0f, tid=%0d, blockid=%0d\n", min_segment, thread_id,
           block_id);
#endif
  }
  return;
}

#endif //}