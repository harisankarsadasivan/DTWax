#ifndef MAIN_PROG
#define MAIN_PROG

#include <assert.h>
#include <cstdint>
#include <iostream>
#include <string>

#include "include/DTW.hpp"
#include "include/binary_IO.hpp"
#include "include/cbf_generator.hpp"
#include "include/common.hpp"
#include "include/hpc_helpers.hpp"

using namespace FullDTW;

#ifdef FP16
#include <cuda_fp16.h>
#define FLOAT2HALF(a) __float2half2_rn(a)
#define HALF2FLOAT(a) __half2float(a)
typedef __half2 value_ht;
#define FP_PIPES 2
#else
#define FP_PIPES 1
#define FLOAT2HALF(a) a
#define HALF2FLOAT(a) a
typedef float value_ht;
#endif

//------------------------------------------------------------time
// macros-----------------------------------------------------//
#define TIMERSTART_CUDA(label)                                                 \
  cudaSetDevice(0);                                                            \
  cudaEvent_t start##label, stop##label;                                       \
  float time##label;                                                           \
  cudaEventCreate(&start##label);                                              \
  cudaEventCreate(&stop##label);                                               \
  cudaEventRecord(start##label, 0);

#define TIMERSTOP_CUDA(label)                                                  \
  cudaSetDevice(0);                                                            \
  cudaEventRecord(stop##label, 0);                                             \
  cudaEventSynchronize(stop##label);                                           \
  cudaEventElapsedTime(&time##label, start##label, stop##label);               \
  std::cout << "TIMING: " << time##label << " ms "                             \
            << ((QUERY_LEN / (time##label * 1e6)) * (REF_LEN)*NUM_READS *      \
                FP_PIPES)                                                      \
            << " GCUPS (" << #label << ")" << std::endl;
//..........................................................other
// macros.......................................................//
#define ASSERT(ans)                                                            \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  }
}
//---------------------------------------------------------global
// vars----------------------------------------------------------//
cudaStream_t stream_var[STREAM_NUM];

int main(int argc, char *argv[]) {

  /* count total cell updates */
  std::cout << "We are going to process "
            << (NUM_READS / 1000000000.0) * QUERY_LEN * REF_LEN
            << " Giga Cell Updates (GCU)" << std::endl;

  // create host storage and buffers on devices
  value_ht *host_query,          // time series on CPU
      *host_dist,                // distance results on CPU
      *host_ref,                 // re-arranged ref  time series on CPU
      *device_query[STREAM_NUM], // time series on GPU
      *device_dist[STREAM_NUM],  // distance results on GPU
      *device_ref;
  raw_t *squiggle_data; // random data generated is stored here.

  //-------------------------------------------------------mem
  // allocation----------------------------------------------------------//
  TIMERSTART(malloc)

  //--------------------------------------------------------host mem
  // allocation--------------------------------------------------//
  ASSERT(cudaMallocHost(&host_query,
                        sizeof(value_ht) * NUM_READS * QUERY_LEN)); /* input */
  ASSERT(cudaMallocHost(&host_ref, sizeof(value_ht) * REF_LEN));    /* input */
  ASSERT(cudaMallocHost(&squiggle_data,
                        sizeof(raw_t) * NUM_READS * QUERY_LEN)); /* input */

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

  //-----------------------------------------------------------squiggle data
  // generation, type conversion, d2h copy target reference and clear some of
  // host mem--------------------------------------//
  TIMERSTART(generate_data)
  generate_cbf(squiggle_data, QUERY_LEN, NUM_READS);
#pragma unroll
  for (uint64_t i = 0; i < (uint64_t)((int64_t)QUERY_LEN * (int64_t)NUM_READS);
       i++) {
    host_query[i] = FLOAT2HALF(squiggle_data[i]);
  }

  //----------re-arranging target reference for memory
  // coalescing-----------------//
  uint64_t k = 0;
  for (uint64_t i = 0; i < SEGMENT_SIZE; i++) {

    for (uint64_t j = 0; j < WARP_SIZE; j++) {
      host_ref[k++] = FLOAT2HALF(squiggle_data[i + (j * SEGMENT_SIZE)]);
    }
  }
  ASSERT(cudaMemcpyAsync(
      device_ref,
      &host_ref[0], //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!//
      sizeof(value_ht) * REF_LEN, cudaMemcpyHostToDevice));
  cudaFreeHost(squiggle_data);
  TIMERSTOP(generate_data)

  /*-------------------------------------------------------------- performs
   * memory I/O and  pairwise DTW
   * computation----------------------------------------- */
  TIMERSTART_CUDA(concurrent_kernel_launch)
  //-------------total batches of concurrent workload to & fro
  // device---------------//
  int batch_count = NUM_READS / (BLOCK_NUM * STREAM_NUM);

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
  ASSERT(cudaDeviceSynchronize());
  TIMERSTOP_CUDA(concurrent_kernel_launch)

  /* -----------------------------------------------------------------print
   * output -----------------------------------------------------*/
#ifdef NV_DEBUG
#ifndef FP16
  for (idxt j = 0; j < NUM_READS; j++) {
    std::cout << HALF2FLOAT(host_dist[j]) << " ";
  }
#else
  for (uint64_t j = 0; j < NUM_READS; j++) {
    std::cout << HALF2FLOAT(host_dist[j].x) << " ";
  }
  std::cout << std::endl;
  for (uint64_t j = 0; j < NUM_READS; j++) {
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

  TIMERSTOP(free)

  return 0;
}

#endif
