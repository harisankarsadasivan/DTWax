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
            << ((num_features + 1) * (num_features + 1) * num_entries *        \
                num_entries) /                                                 \
                   (time##label * 1e6)                                         \
            << " GCUPS (" << #label << ")" << std::endl;
//..................time macros............................//

int main(int argc, char *argv[]) {

  TIMERSTART(malloc)
  index_t num_entries = 1344;  // number of sequences
  index_t num_features = 1024; // length of all sequences

  /* count total cell updates */
  const value_t CU = num_features * num_features * num_entries * num_entries;
  std::cout << "We are going to process " << CU / 1000000000.0
            << " Giga Cell Updates (GCU)" << std::endl;

  // create host storage and buffers on devices
  value_t *data_cpu = nullptr, // time series on CPU
      *dist_cpu = nullptr,     // distance results on CPU
      *data_gpu,               // time series on GPU
      *dist_gpu;               // distance results on GPU
  cudaMallocHost(&data_cpu,
                 sizeof(value_t) * num_entries * num_features); /* input */
  cudaMalloc(&data_gpu, sizeof(value_t) * num_entries * num_features);
  cudaMallocHost(&dist_cpu,
                 sizeof(value_t) * num_entries * num_entries); /* results */
  cudaMalloc(&dist_gpu, sizeof(value_t) * num_entries * num_entries);
  CUERR
  TIMERSTOP(malloc)

  /* load data from memory into CPU array, initialize GPU results */
  TIMERSTART(load_data)
  generate_cbf(data_cpu, num_features, num_entries);
  // load_binary(data_cpu, num_features * num_entries,
  //             "../../../data/kernel/dtw_car.bin");
  cudaMemcpyAsync(data_gpu, data_cpu,
                  sizeof(value_t) * num_features * num_entries,
                  cudaMemcpyHostToDevice);
  CUERR
  cudaMemsetAsync(dist_gpu, 0, sizeof(value_t) * num_entries * num_entries);
  CUERR
  TIMERSTOP(load_data)

  /* perform pairwise DTW computation */
  TIMERSTART_CUDA(computation)
  distances(data_gpu, dist_gpu, num_features, num_entries, (float)0.0);
  CUERR
  TIMERSTOP_CUDA(computation)

  /* copy results to cpu */
  TIMERSTART(save_data)
  cudaMemcpyAsync(dist_cpu, dist_gpu,
                  sizeof(value_t) * num_entries * num_entries,
                  cudaMemcpyDeviceToHost);
  CUERR
  TIMERSTOP(save_data)

#ifdef NV_DEBUG
  /* /1* debug output print *1/ */
  // std::cout << "RESULTS:" << std::endl;
  for (int i = 0; i < num_entries; i++) {
    for (int j = 0; j < num_entries; j++) {
      std::cout << dist_cpu[i * num_entries + j] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
#endif

  TIMERSTART(free)
  cudaFree(data_gpu);
  CUERR
  cudaFree(dist_gpu);
  CUERR
  cudaFreeHost(data_cpu);
  CUERR
  cudaFreeHost(dist_cpu);
  CUERR
  TIMERSTOP(free)

  return 0;
}
