#ifndef FULLDTW
#define FULLDTW

#include "common.hpp"
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
#define ALL 0xFFFFFFFF

template <typename index_t, typename val_t>
__global__ void FullDTW(val_t *subjects, val_t *query, val_t *dist,
                        index_t num_entries, val_t thresh) {

  // cooperative threading
  cg::thread_block_tile<GROUP_SIZE> g =
      cg::tiled_partition<GROUP_SIZE>(cg::this_thread_block());

  /* create vars for indexing */
  const index_t block_id = blockIdx.x;
  const index_t thread_id = cg::this_thread_block().thread_rank();
  const index_t base = 0; // block_id * QUERY_LEN;

  /* initialize penalties */
  val_t penalty_left = INFINITY;
  val_t penalty_diag = INFINITY;
  val_t penalty_here[SEGMENT_SIZE] = {INFINITY};
  val_t penalty_temp[2];

  /* each thread computes CELLS_PER_THREAD adjacent cells, get corresponding sig
   * values */
  val_t subject_val[SEGMENT_SIZE];
  for (int i = 0; i < SEGMENT_SIZE; i++) {
    subject_val[i] = subjects[base + CELLS_PER_THREAD * thread_id + i];
  }

  /* load next WARP_SIZE query values from memory into new_query_val buffer */
  val_t query_val = INFINITY;
  val_t new_query_val = query[block_id * QUERY_LEN + thread_id];

  /* initialize first thread's chunk */
  if (thread_id == 0) {
    query_val = new_query_val;
    penalty_diag = 0;
  }
  new_query_val = __shfl_down_sync(ALL, new_query_val, 1);

  /* calculate full matrix in wavefront parallel manner, multiple cells per
   * thread */
  for (index_t wave = 1; wave <= NUM_WAVES; wave++) {

    /* calculate CELLS_PER_THREAD cells */
    penalty_temp[0] = penalty_here[0];
    penalty_here[0] =
        (query_val - subject_val[0]) * (query_val - subject_val[0]) +
        min(penalty_left, min(penalty_here[0], penalty_diag));

    for (int i = 1; i < SEGMENT_SIZE - 2; i += 2) {
      penalty_temp[1] = penalty_here[i];
      penalty_here[i] =
          (query_val - subject_val[i]) * (query_val - subject_val[i]) +
          min(penalty_here[i - 1], min(penalty_here[i], penalty_temp[0]));

      penalty_temp[0] = penalty_here[i + 1];
      penalty_here[i + 1] =
          (query_val - subject_val[i + 1]) * (query_val - subject_val[i + 1]) +
          min(penalty_here[i - 1], min(penalty_here[i + 1], penalty_temp[1]));
    }

    penalty_here[31] =
        (query_val - subject_val[31]) * (query_val - subject_val[31]) +
        min(penalty_here[30], min(penalty_here[31], penalty_temp[0]));

    /* return result */
    if ((wave >= NUM_WAVES) && (thread_id == RESULT_THREAD_ID)) {
      // printf("@@@result_threadId=%0ld\n",RESULT_THREAD_ID);

      dist[block_id] = penalty_here[RESULT_REG] > thresh ? 0 : 1;

      return;
    }

    /* new_query_val buffer is empty, reload */
    if (wave % WARP_SIZE == 0)
      new_query_val = query[block_id * QUERY_LEN + wave + thread_id];

    /* pass next query_value to each thread */
    query_val = __shfl_up_sync(ALL, query_val, 1);
    if (thread_id == 0)
      query_val = new_query_val;
    new_query_val = __shfl_down_sync(ALL, new_query_val, 1);

    /* transfer border cell info */
    penalty_diag = penalty_left;
    penalty_left = __shfl_up_sync(ALL, penalty_here[SEGMENT_SIZE - 1], 1);
    if (thread_id == 0)
      penalty_left = INFINITY;
  }
}

#endif
