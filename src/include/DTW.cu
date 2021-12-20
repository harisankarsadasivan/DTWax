#ifndef FULLDTW
#define FULLDTW

#define ALL 0xFFFFFFFF

template <typename index_t, typename val_t>
__global__ void FullDTW(val_t *subjects, val_t *query, val_t *dist,
                        index_t num_entries, index_t num_features,
                        val_t thresh) {

  /* create vars for indexing */
  const index_t block_id = blockIdx.x;
  const index_t thread_id = threadIdx.x;
  const index_t base = block_id * num_features;
  const index_t WARP_SIZE = 32;
  const index_t CELLS_PER_THREAD = 32;

  /* initialize penalties */
  val_t penalty_left = INFINITY;
  val_t penalty_diag = INFINITY;
  val_t penalty_here0 = INFINITY;
  val_t penalty_here1 = INFINITY;
  val_t penalty_here2 = INFINITY;
  val_t penalty_here3 = INFINITY;
  val_t penalty_here4 = INFINITY;
  val_t penalty_here5 = INFINITY;
  val_t penalty_here6 = INFINITY;
  val_t penalty_here7 = INFINITY;
  val_t penalty_here8 = INFINITY;
  val_t penalty_here9 = INFINITY;
  val_t penalty_here10 = INFINITY;
  val_t penalty_here11 = INFINITY;
  val_t penalty_here12 = INFINITY;
  val_t penalty_here13 = INFINITY;
  val_t penalty_here14 = INFINITY;
  val_t penalty_here15 = INFINITY;
  val_t penalty_here16 = INFINITY;
  val_t penalty_here17 = INFINITY;
  val_t penalty_here18 = INFINITY;
  val_t penalty_here19 = INFINITY;
  val_t penalty_here20 = INFINITY;
  val_t penalty_here21 = INFINITY;
  val_t penalty_here22 = INFINITY;
  val_t penalty_here23 = INFINITY;
  val_t penalty_here24 = INFINITY;
  val_t penalty_here25 = INFINITY;
  val_t penalty_here26 = INFINITY;
  val_t penalty_here27 = INFINITY;
  val_t penalty_here28 = INFINITY;
  val_t penalty_here29 = INFINITY;
  val_t penalty_here30 = INFINITY;
  val_t penalty_here31 = INFINITY;
  val_t penalty_temp0;
  val_t penalty_temp1;

  /* each thread computes CELLS_PER_THREAD adjacent cells, get corresponding sig
   * values */
  const val_t subject_val0 = subjects[base + CELLS_PER_THREAD * thread_id];
  const val_t subject_val1 = subjects[base + CELLS_PER_THREAD * thread_id + 1];
  const val_t subject_val2 = subjects[base + CELLS_PER_THREAD * thread_id + 2];
  const val_t subject_val3 = subjects[base + CELLS_PER_THREAD * thread_id + 3];
  const val_t subject_val4 = subjects[base + CELLS_PER_THREAD * thread_id + 4];
  const val_t subject_val5 = subjects[base + CELLS_PER_THREAD * thread_id + 5];
  const val_t subject_val6 = subjects[base + CELLS_PER_THREAD * thread_id + 6];
  const val_t subject_val7 = subjects[base + CELLS_PER_THREAD * thread_id + 7];
  const val_t subject_val8 = subjects[base + CELLS_PER_THREAD * thread_id + 8];
  const val_t subject_val9 = subjects[base + CELLS_PER_THREAD * thread_id + 9];
  const val_t subject_val10 =
      subjects[base + CELLS_PER_THREAD * thread_id + 10];
  const val_t subject_val11 =
      subjects[base + CELLS_PER_THREAD * thread_id + 11];
  const val_t subject_val12 =
      subjects[base + CELLS_PER_THREAD * thread_id + 12];
  const val_t subject_val13 =
      subjects[base + CELLS_PER_THREAD * thread_id + 13];
  const val_t subject_val14 =
      subjects[base + CELLS_PER_THREAD * thread_id + 14];
  const val_t subject_val15 =
      subjects[base + CELLS_PER_THREAD * thread_id + 15];
  const val_t subject_val16 =
      subjects[base + CELLS_PER_THREAD * thread_id + 16];
  const val_t subject_val17 =
      subjects[base + CELLS_PER_THREAD * thread_id + 17];
  const val_t subject_val18 =
      subjects[base + CELLS_PER_THREAD * thread_id + 18];
  const val_t subject_val19 =
      subjects[base + CELLS_PER_THREAD * thread_id + 19];
  const val_t subject_val20 =
      subjects[base + CELLS_PER_THREAD * thread_id + 20];
  const val_t subject_val21 =
      subjects[base + CELLS_PER_THREAD * thread_id + 21];
  const val_t subject_val22 =
      subjects[base + CELLS_PER_THREAD * thread_id + 22];
  const val_t subject_val23 =
      subjects[base + CELLS_PER_THREAD * thread_id + 23];
  const val_t subject_val24 =
      subjects[base + CELLS_PER_THREAD * thread_id + 24];
  const val_t subject_val25 =
      subjects[base + CELLS_PER_THREAD * thread_id + 25];
  const val_t subject_val26 =
      subjects[base + CELLS_PER_THREAD * thread_id + 26];
  const val_t subject_val27 =
      subjects[base + CELLS_PER_THREAD * thread_id + 27];
  const val_t subject_val28 =
      subjects[base + CELLS_PER_THREAD * thread_id + 28];
  const val_t subject_val29 =
      subjects[base + CELLS_PER_THREAD * thread_id + 29];
  const val_t subject_val30 =
      subjects[base + CELLS_PER_THREAD * thread_id + 30];
  const val_t subject_val31 =
      subjects[base + CELLS_PER_THREAD * thread_id + 31];

  /* load next WARP_SIZE query values from memory into new_query_val buffer */
  val_t query_val = INFINITY;
  val_t new_query_val = query[thread_id];

  /* initialize first thread's chunk */
  if (thread_id == 0) {
    query_val = new_query_val;
    penalty_diag = 0;
  }
  new_query_val = __shfl_down_sync(ALL, new_query_val, 1);

  /* calculate when to stop, and which thread has final result */
  index_t num_waves = num_features + (num_features - 1) / CELLS_PER_THREAD;
  index_t result_thread_id = (num_features - 1) / CELLS_PER_THREAD;
  index_t result_reg = (num_features - 1) % CELLS_PER_THREAD;

  /* calculate full matrix in wavefront parallel manner, multiple cells per
   * thread */
  for (index_t wave = 1; wave <= num_waves; wave++) {

    /* calculate CELLS_PER_THREAD cells */
    penalty_temp0 = penalty_here0;
    penalty_here0 = (query_val - subject_val0) * (query_val - subject_val0) +
                    min(penalty_left, min(penalty_here0, penalty_diag));
    penalty_temp1 = penalty_here1;
    penalty_here1 = (query_val - subject_val1) * (query_val - subject_val1) +
                    min(penalty_here0, min(penalty_here1, penalty_temp0));
    penalty_temp0 = penalty_here2;
    penalty_here2 = (query_val - subject_val2) * (query_val - subject_val2) +
                    min(penalty_here1, min(penalty_here2, penalty_temp1));
    penalty_temp1 = penalty_here3;
    penalty_here3 = (query_val - subject_val3) * (query_val - subject_val3) +
                    min(penalty_here2, min(penalty_here3, penalty_temp0));
    penalty_temp0 = penalty_here4;
    penalty_here4 = (query_val - subject_val4) * (query_val - subject_val4) +
                    min(penalty_here3, min(penalty_here4, penalty_temp1));
    penalty_temp1 = penalty_here5;
    penalty_here5 = (query_val - subject_val5) * (query_val - subject_val5) +
                    min(penalty_here4, min(penalty_here5, penalty_temp0));
    penalty_temp0 = penalty_here6;
    penalty_here6 = (query_val - subject_val6) * (query_val - subject_val6) +
                    min(penalty_here5, min(penalty_here6, penalty_temp1));
    penalty_temp1 = penalty_here7;
    penalty_here7 = (query_val - subject_val7) * (query_val - subject_val7) +
                    min(penalty_here6, min(penalty_here7, penalty_temp0));
    penalty_temp0 = penalty_here8;
    penalty_here8 = (query_val - subject_val8) * (query_val - subject_val8) +
                    min(penalty_here7, min(penalty_here8, penalty_temp1));
    penalty_temp1 = penalty_here9;
    penalty_here9 = (query_val - subject_val9) * (query_val - subject_val9) +
                    min(penalty_here8, min(penalty_here9, penalty_temp0));
    penalty_temp0 = penalty_here10;
    penalty_here10 = (query_val - subject_val10) * (query_val - subject_val10) +
                     min(penalty_here9, min(penalty_here10, penalty_temp1));
    penalty_temp1 = penalty_here11;
    penalty_here11 = (query_val - subject_val11) * (query_val - subject_val11) +
                     min(penalty_here10, min(penalty_here11, penalty_temp0));
    penalty_temp0 = penalty_here12;
    penalty_here12 = (query_val - subject_val12) * (query_val - subject_val12) +
                     min(penalty_here11, min(penalty_here12, penalty_temp1));
    penalty_temp1 = penalty_here13;
    penalty_here13 = (query_val - subject_val13) * (query_val - subject_val13) +
                     min(penalty_here12, min(penalty_here13, penalty_temp0));
    penalty_temp0 = penalty_here14;
    penalty_here14 = (query_val - subject_val14) * (query_val - subject_val14) +
                     min(penalty_here13, min(penalty_here14, penalty_temp1));
    penalty_temp1 = penalty_here15;
    penalty_here15 = (query_val - subject_val15) * (query_val - subject_val15) +
                     min(penalty_here14, min(penalty_here15, penalty_temp0));
    penalty_temp0 = penalty_here16;
    penalty_here16 = (query_val - subject_val16) * (query_val - subject_val16) +
                     min(penalty_here15, min(penalty_here16, penalty_temp1));
    penalty_temp1 = penalty_here17;
    penalty_here17 = (query_val - subject_val17) * (query_val - subject_val17) +
                     min(penalty_here16, min(penalty_here17, penalty_temp0));
    penalty_temp0 = penalty_here18;
    penalty_here18 = (query_val - subject_val18) * (query_val - subject_val18) +
                     min(penalty_here17, min(penalty_here18, penalty_temp1));
    penalty_temp1 = penalty_here19;
    penalty_here19 = (query_val - subject_val19) * (query_val - subject_val19) +
                     min(penalty_here18, min(penalty_here19, penalty_temp0));
    penalty_temp0 = penalty_here20;
    penalty_here20 = (query_val - subject_val20) * (query_val - subject_val20) +
                     min(penalty_here19, min(penalty_here20, penalty_temp1));
    penalty_temp1 = penalty_here21;
    penalty_here21 = (query_val - subject_val21) * (query_val - subject_val21) +
                     min(penalty_here20, min(penalty_here21, penalty_temp0));
    penalty_temp0 = penalty_here22;
    penalty_here22 = (query_val - subject_val22) * (query_val - subject_val22) +
                     min(penalty_here21, min(penalty_here22, penalty_temp1));
    penalty_temp1 = penalty_here23;
    penalty_here23 = (query_val - subject_val23) * (query_val - subject_val23) +
                     min(penalty_here22, min(penalty_here23, penalty_temp0));
    penalty_temp0 = penalty_here24;
    penalty_here24 = (query_val - subject_val24) * (query_val - subject_val24) +
                     min(penalty_here23, min(penalty_here24, penalty_temp1));
    penalty_temp1 = penalty_here25;
    penalty_here25 = (query_val - subject_val25) * (query_val - subject_val25) +
                     min(penalty_here24, min(penalty_here25, penalty_temp0));
    penalty_temp0 = penalty_here26;
    penalty_here26 = (query_val - subject_val26) * (query_val - subject_val26) +
                     min(penalty_here25, min(penalty_here26, penalty_temp1));
    penalty_temp1 = penalty_here27;
    penalty_here27 = (query_val - subject_val27) * (query_val - subject_val27) +
                     min(penalty_here26, min(penalty_here27, penalty_temp0));
    penalty_temp0 = penalty_here28;
    penalty_here28 = (query_val - subject_val28) * (query_val - subject_val28) +
                     min(penalty_here27, min(penalty_here28, penalty_temp1));
    penalty_temp1 = penalty_here29;
    penalty_here29 = (query_val - subject_val29) * (query_val - subject_val29) +
                     min(penalty_here28, min(penalty_here29, penalty_temp0));
    penalty_temp0 = penalty_here30;
    penalty_here30 = (query_val - subject_val30) * (query_val - subject_val30) +
                     min(penalty_here29, min(penalty_here30, penalty_temp1));
    penalty_here31 = (query_val - subject_val31) * (query_val - subject_val31) +
                     min(penalty_here30, min(penalty_here31, penalty_temp0));

    /* return result */
    if (wave >= num_waves) {
      // printf("@@@result_threadId=%0ld\n",result_thread_id);
      if (thread_id == result_thread_id) {
        switch (result_reg) {
        case 0:
          dist[block_id] = penalty_here0 > thresh ? 0 : 1;
          break;
        case 1:
          dist[block_id] = penalty_here1 > thresh ? 0 : 1;
          break;
        case 2:
          dist[block_id] = penalty_here2 > thresh ? 0 : 1;
          break;
        case 3:
          dist[block_id] = penalty_here3 > thresh ? 0 : 1;
          break;
        case 4:
          dist[block_id] = penalty_here4 > thresh ? 0 : 1;
          break;
        case 5:
          dist[block_id] = penalty_here5 > thresh ? 0 : 1;
          break;
        case 6:
          dist[block_id] = penalty_here6 > thresh ? 0 : 1;
          break;
        case 7:
          dist[block_id] = penalty_here7 > thresh ? 0 : 1;
          break;
        case 8:
          dist[block_id] = penalty_here8 > thresh ? 0 : 1;
          break;
        case 9:
          dist[block_id] = penalty_here9 > thresh ? 0 : 1;
          break;
        case 10:
          dist[block_id] = penalty_here10 > thresh ? 0 : 1;
          break;
        case 11:
          dist[block_id] = penalty_here11 > thresh ? 0 : 1;
          break;
        case 12:
          dist[block_id] = penalty_here12 > thresh ? 0 : 1;
          break;
        case 13:
          dist[block_id] = penalty_here13 > thresh ? 0 : 1;
          break;
        case 14:
          dist[block_id] = penalty_here14 > thresh ? 0 : 1;
          break;
        case 15:
          dist[block_id] = penalty_here15 > thresh ? 0 : 1;
          break;
        case 16:
          dist[block_id] = penalty_here16 > thresh ? 0 : 1;
          break;
        case 17:
          dist[block_id] = penalty_here17 > thresh ? 0 : 1;
          break;
        case 18:
          dist[block_id] = penalty_here18 > thresh ? 0 : 1;
          break;
        case 19:
          dist[block_id] = penalty_here19 > thresh ? 0 : 1;
          break;
        case 20:
          dist[block_id] = penalty_here20 > thresh ? 0 : 1;
          break;
        case 21:
          dist[block_id] = penalty_here21 > thresh ? 0 : 1;
          break;
        case 22:
          dist[block_id] = penalty_here22 > thresh ? 0 : 1;
          break;
        case 23:
          dist[block_id] = penalty_here23 > thresh ? 0 : 1;
          break;
        case 24:
          dist[block_id] = penalty_here24 > thresh ? 0 : 1;
          break;
        case 25:
          dist[block_id] = penalty_here25 > thresh ? 0 : 1;
          break;
        case 26:
          dist[block_id] = penalty_here26 > thresh ? 0 : 1;
          break;
        case 27:
          dist[block_id] = penalty_here27 > thresh ? 0 : 1;
          break;
        case 28:
          dist[block_id] = penalty_here28 > thresh ? 0 : 1;
          break;
        case 29:
          dist[block_id] = penalty_here29 > thresh ? 0 : 1;
          break;
        case 30:
          dist[block_id] = penalty_here30 > thresh ? 0 : 1;
          break;
        case 31:
          dist[block_id] = penalty_here31 > thresh ? 0 : 1;
          break;
        }
      }
      return;
    }

    /* new_query_val buffer is empty, reload */
    if (wave % WARP_SIZE == 0)
      new_query_val = query[wave + thread_id];

    /* pass next query_value to each thread */
    query_val = __shfl_up_sync(ALL, query_val, 1);
    if (thread_id == 0)
      query_val = new_query_val;
    new_query_val = __shfl_down_sync(ALL, new_query_val, 1);

    /* transfer border cell info */
    penalty_diag = penalty_left;
    penalty_left = __shfl_up_sync(ALL, penalty_here31, 1);
    if (thread_id == 0)
      penalty_left = INFINITY;
  }
}

#endif
