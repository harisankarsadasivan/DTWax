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
//------------------time macros--------------------------//
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
            << ((QUERY_LEN) * (REF_LEN)*num_entries * FP_PIPES) /              \
                   (time##label * 1e6)                                         \
            << " GCUPS (" << #label << ")" << std::endl;
//..................time macros............................//
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

cudaStream_t stream_var[STREAM_NUM];

int main(int argc, char *argv[]) {

  index_t num_entries = BLOCK_NUM; // number of sequences

  /* count total cell updates */
  const value_t CU = QUERY_LEN * REF_LEN * num_entries;
  std::cout << "We are going to process " << CU / 1000000000.0
            << " Giga Cell Updates (GCU)" << std::endl;

  // create host storage and buffers on devices
  value_ht *host_query,          // time series on CPU
      *host_dist,                // distance results on CPU
      *device_query[STREAM_NUM], // time series on GPU
      *device_dist[STREAM_NUM],  // distance results on GPU
      *host_ref, *device_ref;
  raw_t *squiggle_data; // random data generated is stored here.
  //------mem allocation---------------//
  TIMERSTART(malloc)
  //--------host mem allocation-----------------//
  ASSERT(cudaMallocHost(&host_query, sizeof(value_ht) * num_entries *
                                         QUERY_LEN)); /* input */
  ASSERT(cudaMallocHost(&squiggle_data,
                        sizeof(raw_t) * num_entries * QUERY_LEN)); /* input */
  ASSERT(cudaMallocHost(&host_ref, sizeof(value_ht) * REF_LEN));   /* input */
  ASSERT(
      cudaMallocHost(&host_dist, sizeof(value_ht) * num_entries)); /* results */

  //-------dev mem allocation----------//
  ASSERT(cudaMalloc(&device_ref, sizeof(value_ht) * REF_LEN));
  for (int stream_id = 0; stream_id < STREAM_NUM; stream_id++) {
    ASSERT(cudaMalloc(&device_query[stream_id],
                      sizeof(value_ht) * num_entries * QUERY_LEN / STREAM_NUM));
    ASSERT(cudaMalloc(&device_dist[stream_id],
                      sizeof(value_ht) * num_entries / STREAM_NUM));
    ASSERT(cudaStreamCreate(&stream_var[stream_id]));
  }

  TIMERSTOP(malloc)

  //--------data generation and type conversion-------------------//
  TIMERSTART(generate_data)
  generate_cbf(squiggle_data, QUERY_LEN, num_entries);
#pragma unroll
  for (int i = 0; i < (QUERY_LEN * num_entries); i++) {
    host_query[i] = FLOAT2HALF(squiggle_data[i]);
  }
  TIMERSTOP(generate_data)
  /* load data from memory into CPU array, initialize GPU results */
  TIMERSTART(load_data)

  // load_binary(host_dist, QUERY_LEN * num_entries,
  //             "../../../data/kernel/dtw_car.bin");
  ASSERT(cudaMemcpyAsync(
      device_ref,
      &host_query[0], //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!//
      sizeof(value_ht) * REF_LEN, cudaMemcpyHostToDevice));

  for (int stream_id = 0; stream_id < STREAM_NUM; stream_id++) {

    ASSERT(cudaMemcpyAsync(
        device_query[stream_id],
        &host_query[stream_id * QUERY_LEN * num_entries / STREAM_NUM],
        sizeof(value_ht) * QUERY_LEN * num_entries / STREAM_NUM,
        cudaMemcpyHostToDevice, stream_var[stream_id]));
    // CUERR cudaMemsetAsync(&device_dist[stream_id], 0, (sizeof(value_ht) *
    // num_entries/STREAM_NUM),stream_var[stream_id]);
  }

  TIMERSTOP(load_data)

  /* perform pairwise DTW computation */
  TIMERSTART_CUDA(computation)
  for (int stream_id = 0; stream_id < STREAM_NUM; stream_id++) {
    distances<value_ht, index_t>(
        device_ref, device_query[stream_id], device_dist[stream_id],
        num_entries / STREAM_NUM, FLOAT2HALF(0), stream_var[stream_id]);
  }
  ASSERT(cudaDeviceSynchronize());
  TIMERSTOP_CUDA(computation)

  /* copy results to cpu */
  TIMERSTART(save_data)
  for (int stream_id = 0; stream_id < STREAM_NUM; stream_id++) {
    ASSERT(cudaMemcpyAsync(&host_dist[stream_id * num_entries / STREAM_NUM],
                           device_dist[stream_id],
                           sizeof(value_ht) * num_entries / STREAM_NUM,
                           cudaMemcpyDeviceToHost, stream_var[stream_id]));
  }
  TIMERSTOP(save_data)

#ifdef NV_DEBUG
#ifndef FP16
  for (idxt j = 0; j < num_entries; j++) {
    std::cout << HALF2FLOAT(host_dist[j]) << " ";
  }
#else
  for (idxt j = 0; j < num_entries; j++) {
    std::cout << HALF2FLOAT(host_dist[j].x) << " ";
  }
  std::cout << std::endl;
  for (idxt j = 0; j < num_entries; j++) {
    std::cout << HALF2FLOAT(host_dist[j].y) << " ";
  }

#endif
  std::cout << std::endl;
#endif

  TIMERSTART(free)
  cudaFree(device_dist);

  cudaFree(device_query);

  cudaFree(device_ref);

  cudaFreeHost(host_ref);

  cudaFreeHost(host_query);

  cudaFreeHost(host_dist);

  cudaFreeHost(squiggle_data);
  TIMERSTOP(free)

  return 0;
}

#endif