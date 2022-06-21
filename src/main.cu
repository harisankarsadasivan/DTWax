#ifndef MAIN_PROG
#define MAIN_PROG

#include <assert.h>
#include <cstdint>
#include <iostream>
#include <string>

#include "include/DTW.hpp"
#include "include/binary_IO.hpp"
#include "include/common.hpp"
#include "include/generate_load_squiggle.hpp"
#include "include/hpc_helpers.hpp"
#include "include/normalizer.cu"

using namespace FullDTW;

//---------------------------------------------------------global
// vars----------------------------------------------------------//
cudaStream_t stream_var[STREAM_NUM];

int main(int argc, char **argv) {

  // create host storage and buffers on devices
  value_ht *host_query,          // time series on CPU
      *host_dist,                // distance results on CPU
      *host_ref,                 // re-arranged ref  time series on CPU
      *device_query[STREAM_NUM], // time series on GPU
      *device_dist[STREAM_NUM],  // distance results on GPU
      *device_ref;

  raw_t *raw_array = NULL;

  TIMERSTART(time_until_data_is_loaded)

  //*************************************************************LOAD FROM
  // FILE********************************************************//
  index_t NUM_READS; // counter to count number of reads to be processed
  squiggle_loader *loader = new squiggle_loader;
  loader->load_data(argv[1], raw_array,
                    NUM_READS); // load from input ONT data folder with FAST5
  ASSERT(cudaMallocHost(
      &raw_array,
      (sizeof(raw_t) *
       (NUM_READS * QUERY_LEN)))); // host pinned memory for raw data from FAST5

  loader->load_query(raw_array);

  delete loader;

  //****************************************************NORMALIZER****************************************//
  // normalizer instance - does h2h pinned mem transfer, CUDNN setup and zscore
  // normalization, normalized raw_t output is returned in same array as input
  normalizer *NMZR = new normalizer;
  TIMERSTART(normalizer_kernel)
  NMZR->normalize(raw_array, NUM_READS);
  TIMERSTOP(normalizer_kernel)
  std::cout << "Normalizer processed  " << (QUERY_LEN * NUM_READS)
            << " raw samples in this time\n";

#ifdef NV_DEBUG
  NMZR->print_normalized_query(raw_array, NUM_READS);
#endif

  delete NMZR;
  // normalizartion completed

  //****************************************************FLOAT to
  //__half2****************************************//
  ASSERT(cudaMallocHost(&host_query,
                        sizeof(value_ht) * NUM_READS * QUERY_LEN)); /* input */
  std::cout << "Normalized data:\n";
  for (index_t i = 0; i < NUM_READS; i++) {
    for (index_t j = 0; j < QUERY_LEN; j++) {
      host_query[(i * NUM_READS + j)] =
          FLOAT2HALF(raw_array[(i * NUM_READS + j)]);
    }
  }
  cudaFreeHost(raw_array);
  TIMERSTOP(time_until_data_is_loaded)

  //****************************************************MEM
  // allocation****************************************//
  TIMERSTART(malloc)
  //--------------------------------------------------------host mem
  // allocation--------------------------------------------------//

  ASSERT(cudaMallocHost(&host_ref, sizeof(value_ht) * REF_LEN)); /* input */

  ASSERT(
      cudaMallocHost(&host_dist, sizeof(value_ht) * NUM_READS)); /* results */

  //-------------------------------------------------------------dev mem
  // allocation-------------------------------------------------//
  ASSERT(cudaMalloc(&device_ref, sizeof(value_ht) * REF_LEN));

  for (int stream_id = 0; stream_id < STREAM_NUM; stream_id++) {
    ASSERT(cudaMalloc(&device_query[stream_id],
                      (sizeof(value_ht) * BLOCK_NUM * QUERY_LEN)));
    ASSERT(cudaMalloc(&device_dist[stream_id], sizeof(value_ht) * BLOCK_NUM));
    ASSERT(cudaStreamCreate(&stream_var[stream_id]));
  }

  TIMERSTOP(malloc)

  //****************************************************Target ref
  // re-organization for better mem coalescing & target
  // loading****************************************//

  TIMERSTART(load_target)
  uint64_t k = 0;
#pragma omp parallel for
  for (index_t i = 0; i < SEGMENT_SIZE; i++) {

    for (index_t j = 0; j < WARP_SIZE; j++) {
      host_ref[k] = host_query[i + (j * SEGMENT_SIZE)];
      // std::cout << HALF2FLOAT(host_ref[k].x) << ",";
      k++;
    }
  }

  ASSERT(cudaMemcpyAsync(
      device_ref,
      &host_ref[0], //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!//
      sizeof(value_ht) * REF_LEN, cudaMemcpyHostToDevice));

  TIMERSTOP(load_target)

  //****************************************************Mem I/O and DTW
  // computation****************************************//
  TIMERSTART_CUDA(concurrent_DTW_kernel_launch)
  //-------------total batches of concurrent workload to & fro
  // device---------------//
  int batch_count = NUM_READS / (BLOCK_NUM * STREAM_NUM);

  if (batch_count > 0) {
    for (int batch_id = 0; batch_id < batch_count; batch_id++) {
      for (int stream_id = 0; stream_id < STREAM_NUM; stream_id++) {
        //----h2d copy-------------//
        ASSERT(cudaMemcpyAsync(
            device_query[stream_id],
            &host_query[(batch_id * STREAM_NUM * QUERY_LEN * BLOCK_NUM) +
                        (stream_id * QUERY_LEN * BLOCK_NUM)],
            sizeof(value_ht) * QUERY_LEN * BLOCK_NUM, cudaMemcpyHostToDevice,
            stream_var[stream_id]));

        //---------launch kernels------------//
        distances<value_ht, index_t>(device_ref, device_query[stream_id],
                                     device_dist[stream_id], BLOCK_NUM,
                                     FLOAT2HALF(0), stream_var[stream_id]);

        //-----d2h copy--------------//
        ASSERT(cudaMemcpyAsync(&host_dist[(batch_id * STREAM_NUM * BLOCK_NUM) +
                                          (stream_id * BLOCK_NUM)],
                               device_dist[stream_id],
                               sizeof(value_ht) * BLOCK_NUM,
                               cudaMemcpyDeviceToHost, stream_var[stream_id]));
      }
    }
  } else {

    //----h2d copy-------------//
    ASSERT(cudaMemcpyAsync(device_query[0], &host_query[0],
                           sizeof(value_ht) * QUERY_LEN * NUM_READS,
                           cudaMemcpyHostToDevice, stream_var[0]));

    //---------launch kernels------------//
    distances<value_ht, index_t>(device_ref, device_query[0], device_dist[0],
                                 NUM_READS, FLOAT2HALF(0), stream_var[0]);

    //-----d2h copy--------------//
    ASSERT(cudaMemcpyAsync(&host_dist[0], device_dist[0],
                           sizeof(value_ht) * NUM_READS, cudaMemcpyDeviceToHost,
                           stream_var[0]));
  }
  ASSERT(cudaDeviceSynchronize());
  TIMERSTOP_CUDA(concurrent_DTW_kernel_launch)

  /* -----------------------------------------------------------------print
   * output -----------------------------------------------------*/
#ifdef NV_DEBUG
#ifndef FP16
  for (index_t j = 0; j < NUM_READS; j++) {
    std::cout << HALF2FLOAT(host_dist[j]) << " ";
  }
#else
  for (index_t j = 0; j < NUM_READS; j++) {
    std::cout << HALF2FLOAT(host_dist[j].x) << " ";
  }
  std::cout << std::endl;
  for (index_t j = 0; j < NUM_READS; j++) {
    std::cout << HALF2FLOAT(host_dist[j].y) << " ";
  }

#endif
  std::cout << std::endl;
#endif

  /* -----------------------------------------------------------------free
   * memory -----------------------------------------------------*/
  TIMERSTART(free)
  for (int stream_id = 0; stream_id < STREAM_NUM; stream_id++) {
    cudaFree(device_dist[stream_id]);
    cudaFree(device_query[stream_id]);
  }
  cudaFree(device_ref);
  cudaFreeHost(host_query);
  cudaFreeHost(host_dist);
  cudaFreeHost(host_ref);
  TIMERSTOP(free)

  return 0;
}

#endif
