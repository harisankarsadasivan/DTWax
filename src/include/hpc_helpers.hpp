#ifndef HPC_HELPERS_HPP
#define HPC_HELPERS_HPP

#include <cstdint>
#include <iostream>
#include <stdio.h>
#include <string>
#include <unistd.h>

#ifndef __CUDACC__
#include <chrono>
#endif

#ifndef __CUDACC__
#define TIMERSTART(label)                                                      \
  std::chrono::time_point<std::chrono::system_clock> a##label, b##label;       \
  a##label = std::chrono::system_clock::now();
#else
#define TIMERSTART(label)                                                      \
  cudaSetDevice(0);                                                            \
  cudaEvent_t start##label, stop##label;                                       \
  float time##label;                                                           \
  cudaEventCreate(&start##label);                                              \
  cudaEventCreate(&stop##label);                                               \
  cudaEventRecord(start##label, 0);
#endif

#ifndef __CUDACC__
#define TIMERSTOP(label)                                                       \
  b##label = std::chrono::system_clock::now();                                 \
  std::chrono::duration<double> delta##label = b##label - a##label;            \
  std::cout << "# elapsed time (" << #label << "): " << delta##label.count()   \
            << "s" << std::endl;
#else
#define TIMERSTOP(label)                                                       \
  cudaSetDevice(0);                                                            \
  cudaEventRecord(stop##label, 0);                                             \
  cudaEventSynchronize(stop##label);                                           \
  cudaEventElapsedTime(&time##label, start##label, stop##label);               \
  std::cout << "TIMING: " << time##label << " ms (" << #label << ")"           \
            << std::endl;
#endif

#ifdef __CUDACC__
#define CUERR                                                                  \
  {                                                                            \
    cudaError_t err;                                                           \
    if ((err = cudaGetLastError()) != cudaSuccess) {                           \
      std::cout << "CUDA error: " << cudaGetErrorString(err) << " : "          \
                << __FILE__ << ", line " << __LINE__ << std::endl;             \
      exit(1);                                                                 \
    }                                                                          \
  }
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

#define TIMERSTOP_CUDA(label, NUM_READS)                                       \
  cudaSetDevice(0);                                                            \
  cudaEventRecord(stop##label, 0);                                             \
  cudaEventSynchronize(stop##label);                                           \
  cudaEventElapsedTime(&time##label, start##label, stop##label);               \
  std::cout << "TIMING: " << time##label << " ms "                             \
            << ((QUERY_LEN / (time##label * 1e3)) * NUM_READS / 10)            \
            << " Mbps (" << #label << ")" << std::endl;
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
// safe division
#define SDIV(x, y) (((x) + (y)-1) / (y))

// void input_parser(int argc, char *argv[], std::string &ip_path,
//                   std::string &model_file, std::string &ref_file) {
//   // int opt;
//   // std::cout << argc;
//   // while ((opt = getopt(argc, argv, “abc :”)) != -1) {
//   //   switch (opt) {
//   //   case ‘i’:
//   //     ip_path = optarg;
//   //     std::cout << "input path is " << ip_path << "\n";
//   //     break;
//   //   case ‘k’:
//   //     model_file = optarg;
//   //     std::cout << "model file is " << model_file << "\n";
//   //     break;
//   //   case ‘r’:
//   //     ref_file = optarg;
//   //     std::cout << "fasta reference file is " << ref_file << "\n";
//   //     break;
//   //   }
//   // }
// }
#endif