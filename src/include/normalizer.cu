#ifndef NORMALIZER
#define NORMALIZER

#include "common.hpp"
#include <cuda_runtime.h>
#include <cudnn.h>

#define CUDNN_ASSERT(func)                                                     \
  {                                                                            \
    cudnnStatus_t e = (func);                                                  \
    std::cout << "\ncuDTW::  cuDNN Normalizer returned: "                      \
              << cudnnGetErrorString(e) << "\n";                               \
  }

// normalizer class
class normalizer {
public:
  void normalize(raw_t *raw_squiggle_array, index_t num_reads, index_t length);
  normalizer(idxt NUM_READS); // CUDNN normalizer
  ~normalizer();
  void print_normalized_query(raw_t *raw_array, index_t NUM_READS,
                              std::vector<std::string> &read_ids);

private:
  float *bnScale, *bnBias, *bnScale_h, *bnBias_h;
  idxt NUM_READS;
  float alpha[1] = {1};
  float beta[1] = {0.0};
};

void normalizer::print_normalized_query(raw_t *raw_array, index_t NUM_READS,
                                        std::vector<std::string> &read_ids) {
  std::cout << "Normalized query:\n";
  for (index_t i = 0; i < NUM_READS; i++) {
    std::cout << "cuDTW:: " << read_ids[i] << "\n";
    for (index_t j = 0; j < QUERY_LEN; j++) {
      std::cout << raw_array[(i * QUERY_LEN + j)] << ",";
    }
    std::cout << "\n";
  }
  std::cout << "\n=================\n";
}
normalizer::~normalizer() {
  cudaFree(bnScale);
  cudaFree(bnBias);
  cudaFreeHost(bnScale_h);
  cudaFreeHost(bnBias_h);
}
normalizer::normalizer(idxt NUM_READS) {

  // create scale and bias vectors
  cudaMallocHost(&bnScale_h, (QUERY_LEN * sizeof(float) * NUM_READS));
  cudaMallocHost(&bnBias_h, (QUERY_LEN * sizeof(float) * NUM_READS));
  for (int i = 0; i < QUERY_LEN * NUM_READS; i++) {
    bnScale_h[i] = 1.0f;
    bnBias_h[i] = 0.0f;
  }
  cudaMalloc(&bnScale, (QUERY_LEN * sizeof(float) * NUM_READS));
  cudaMalloc(&bnBias, (QUERY_LEN * sizeof(float) * NUM_READS));

  cudaMemcpyAsync(bnScale,
                  &bnScale_h[0], //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!//
                  sizeof(float) * NUM_READS * QUERY_LEN,
                  cudaMemcpyHostToDevice);
  cudaMemcpyAsync(bnBias,
                  &bnBias_h[0], //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!//
                  sizeof(float) * NUM_READS * QUERY_LEN,
                  cudaMemcpyHostToDevice);
}

__inline__ void normalizer::normalize(raw_t *raw_squiggle_array,
                                      index_t num_reads, index_t length) {

  int c = num_reads, h = length; // nchw format for cudnn

  raw_t *x; // output,input array

  cudaMalloc(&x, (sizeof(raw_t) * c * h));

  cudaMemcpy(x, &raw_squiggle_array[0], (sizeof(raw_t) * c * h),
             cudaMemcpyHostToDevice);

  cudnnHandle_t handle_;
  cudnnCreate(&handle_);
  cudnnDataType_t dtype = CUDNN_DATA_FLOAT;
  cudnnTensorFormat_t format = CUDNN_TENSOR_NCHW; // CUDNN_TENSOR_NCHW;
  cudnnBatchNormMode_t mode = CUDNN_BATCHNORM_SPATIAL;

  // descriptors
  cudnnTensorDescriptor_t x_desc, bnScaleBiasMeanVarDesc;
  cudnnCreateTensorDescriptor(&x_desc);
  cudnnSetTensor4dDescriptor(x_desc, format, dtype, 1, c, h, 1);

  cudnnCreateTensorDescriptor(&bnScaleBiasMeanVarDesc);
  cudnnDeriveBNTensorDescriptor(bnScaleBiasMeanVarDesc, x_desc, mode);

  // normalize
  cudnnBatchNormalizationForwardTraining(
      handle_, mode, alpha, beta, x_desc, x, x_desc, x, bnScaleBiasMeanVarDesc,
      bnScale, bnBias, 1.0 / (1.0 + h), NULL, NULL, 0.0001f, NULL, NULL);

  cudaMemcpy(

      &raw_squiggle_array[0], x, (sizeof(raw_t) * c * h),
      cudaMemcpyDeviceToHost);

  // std::cout << "cudnn normalized output:\n";
  // for (uint64_t i = 0; i < (c * h); i++) {

  //   std::cout << raw_squiggle_array[i] << ",";
  // }
  // cudnnDestroy(handle_);
  return;
}

#endif