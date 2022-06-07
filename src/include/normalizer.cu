#ifndef NORMALIZER
#define NORMALIZER

#include <cuda_runtime.h>
#include <cudnn.h>

__inline__ void normalize(raw_t *raw_squiggle_array, const void *bnScale,
                          const void *bnBias) {

  int c = NUM_READS, h = QUERY_LEN; // nchw format for cudnn
  float alpha[1] = {1};
  float beta[1] = {0.0};
  cudnnStatus_t status;

  raw_t *y, *x; // output,input array
  cudaMalloc(&y, sizeof(raw_t) * NUM_READS * QUERY_LEN);
  cudaMalloc(&x, sizeof(raw_t) * NUM_READS * QUERY_LEN);
  cudaMemcpy(
      x,
      &raw_squiggle_array[0], //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!//
      sizeof(raw_t) * NUM_READS * QUERY_LEN, cudaMemcpyHostToDevice);

  cudnnHandle_t handle_;
  cudnnCreate(&handle_);
  cudnnDataType_t dtype = CUDNN_DATA_FLOAT;
  cudnnTensorFormat_t format = CUDNN_TENSOR_NHWC; // CUDNN_TENSOR_NCHW;
  cudnnBatchNormMode_t mode = CUDNN_BATCHNORM_PER_ACTIVATION;

  // descriptors
  cudnnTensorDescriptor_t x_desc, y_desc, bnScaleBiasMeanVarDesc;
  cudnnCreateTensorDescriptor(&x_desc);
  cudnnSetTensor4dDescriptor(x_desc, format, dtype, 1, c, h, 1);

  cudnnCreateTensorDescriptor(&y_desc);
  cudnnSetTensor4dDescriptor(y_desc, format, dtype, 1, c, h, 1);

  cudnnCreateTensorDescriptor(&bnScaleBiasMeanVarDesc);
  cudnnDeriveBNTensorDescriptor(bnScaleBiasMeanVarDesc, x_desc, mode);
  //   cudnnSetTensor4dDescriptor(bnScaleBiasMeanVarDesc, format, dtype, 1, c,
  //   h, 1);

  // normalize
  status = cudnnBatchNormalizationForwardTraining(
      handle_, mode, alpha, beta, x_desc, x, y_desc, y, bnScaleBiasMeanVarDesc,
      bnScale, bnBias, 1.0 / (1.0 + h), NULL, NULL, 0.0001f, NULL, NULL);
  cudaDeviceSynchronize();
  std::cout << "cudann status: " << status << "\n";
  cudaMemcpy(

      &raw_squiggle_array[0], y, sizeof(raw_t) * NUM_READS * QUERY_LEN,
      cudaMemcpyDeviceToHost);

  std::cout << "cudnn normalized output:\n";
  for (uint64_t i = 0; i < (uint64_t)((int64_t)QUERY_LEN * (int64_t)NUM_READS);
       i++) {

    std::cout << raw_squiggle_array[i] << ",";
  }
  cudnnDestroy(handle_);
  cudaFree(y);
  cudaFree(x);
  return;
}
//#include "stdio.h"
// #include <thrust/device_vector.h>
// #include <thrust/reduce.h>

// #define FMA_fp32(a, b, c) __fmaf_ieee_rn(a, b, c)
// #define ADD_fp32(a, b) __fmaf_ieee_rn(a, 1, b)
// #define DIV_fp32(a, b) __fdiv_rn(a, b) // make sure b is power of 2
// #define SQRT_fp32(a) sqrtf(a)          // a is to be float

// // warp sum reduction using shuffle operation
// __inline__ __device__ raw_t warpReduceSum(raw_t val) {
//   for (int mask = warpSize / 2; mask > 0; mask /= 2)
//     val = ADD_fp32(
//         val, __shfl_down_sync(
//                  0xffffffff, val, mask,
//                  WARP_SIZE)); // butterfly reduction copies sum to all
//                  threads
//   return val;
// }

// // block sum
// __inline__ __device__ raw_t blockReduceSum(raw_t val) {

//   static __shared__ raw_t shared[32]; // Shared mem for 32 partial sums
//   int lane = (threadIdx.x & (WARP_SIZE - 1));
//   int wid = threadIdx.x >> LOG_WARP_SIZE;

//   val = warpReduceSum(val); // Each warp performs partial reduction

//   if (lane == 0)
//     shared[wid] = val; // Write reduced value to shared memory

//   __syncthreads(); // Wait for all partial reductions

//   // read from shared memory only if that warp existed
//   val = (threadIdx.x < (blockDim.x >> LOG_WARP_SIZE)) ? shared[lane] : (0);

//   if (wid == 0) {
//     val = warpReduceSum(val); // Final reduce within first warp
//     // val = __shfl_sync(0xffffffff, val, 0, WARP_SIZE);
//     // shared[threadIdx.x] = val;
//     shared[threadIdx.x] = val;
//   }

//   __syncthreads();
//   val = shared[0];
//   return val;
// }

// // normalization kernel definition
// __global__ void deviceReduceKernel(raw_t *raw_squiggle_array) {
//   raw_t sum = (0),
//         sum2 =
//             (0); // sum is N*E[X], sum2 is E[X^2] where N is sample size and
//                  // X is random variable. Std dev =
//                  SQRT_fp32(E[X^2]-(E[X])^2)

//   sum = raw_squiggle_array[blockIdx.x * blockDim.x + threadIdx.x];
//   sum2 = FMA_fp32(sum, sum, (0));
//   // printf("1.sum=%0f\n", sum);
//   // printf("1.sum2=%0f\n", sum2);

//   sum = blockReduceSum(sum);
//   sum2 = blockReduceSum(sum2);

//   // printf("2.sum=%0f\n", sum);
//   // printf("2.sum2=%0f\n", sum2);

//   // printf("3.sum=%0f\n", sum);
//   // printf("3.sum2=%0f\n", sum2);

//   sum = DIV_fp32(sum, QUERY_LEN); // mean
//   sum2 = DIV_fp32(sum2, QUERY_LEN);
//   sum2 = SQRT_fp32(FMA_fp32(FMA_fp32(sum, sum, (0)), (-1),
//                             sum2)); // stdev
//   // printf("mean=%0f\n", sum);
//   // printf("stdev=%0f\n", sum2);
//   for (index_t i = blockDim.x * blockIdx.x + threadIdx.x;
//        i < (gridDim.x * QUERY_LEN); i += blockDim.x) {
//     raw_squiggle_array[i] = DIV_fp32(FMA_fp32(sum, -1,
//     raw_squiggle_array[i]),
//                                      sum2); // this is the normalized output
//     // printf("raw_squiggle_array[%0d]=%0f\n", i, raw_squiggle_array[i]);
//   }
// }

// template <typename T> struct square {
//   __host__ __device__ T operator()(const T &x) const { return x * x; }
// };

// template <typename T>
// void mean_and_var(T a, int n, double *p_mean, double *p_var) {
//   double sum = thrust::reduce(a, &a[n], 0.0, thrust::plus<double>());
//   double sum_square = thrust::transform_reduce(a, &a[n], square<double>(),
//   0.0,
//                                                thrust::plus<double>());
//   double mean = sum / n;
//   *p_mean = mean;
//   *p_var = (sum_square / n) - mean * mean;

#endif