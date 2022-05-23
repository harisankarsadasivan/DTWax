#ifndef NORMALIZER
#define NORMALIZER

#define FMA_fp32(a, b, c) __fmaf_ieee_rn(a, b, c)
#define ADD_fp32(a, b) __fmaf_ieee_rn(a, 1, b)
#define DIV_fp32(a, b) __fdiv_rn(a, b) // make sure b is power of 2
#define SQRT_fp32(a) sqrtf(a)          // a is to be float

// warp sum reduction using shuffle operation
__inline__ __device__ raw_t warpReduceSum(raw_t val) {
  for (int mask = warpSize / 2; mask > 0; mask /= 2)
    val = ADD_fp32(
        val, __shfl_down_sync(
                 0xffffffff, val, mask,
                 WARP_SIZE)); // butterfly reduction copies sum to all threads
  return val;
}

// block sum
__inline__ __device__ raw_t blockReduceSum(raw_t val) {

  static __shared__ raw_t shared[32]; // Shared mem for 32 partial sums
  int lane = (threadIdx.x & (WARP_SIZE - 1));
  int wid = threadIdx.x >> LOG_WARP_SIZE;

  val = warpReduceSum(val); // Each warp performs partial reduction

  if (lane == 0)
    shared[wid] = val; // Write reduced value to shared memory

  __syncthreads(); // Wait for all partial reductions

  // read from shared memory only if that warp existed
  val = (threadIdx.x < (blockDim.x >> LOG_WARP_SIZE)) ? shared[lane] : (0);

  if (wid == 0) {
    val = warpReduceSum(val); // Final reduce within first warp
    // val = __shfl_sync(0xffffffff, val, 0, WARP_SIZE);
    // shared[threadIdx.x] = val;
    shared[threadIdx.x] = val;
  }

  __syncthreads();
  val = shared[0];
  return val;
}

// normalization kernel definition
__global__ void deviceReduceKernel(raw_t *raw_squiggle_array) {
  raw_t sum = (0),
        sum2 =
            (0); // sum is N*E[X], sum2 is E[X^2] where N is sample size and
                 // X is random variable. Std dev = SQRT_fp32(E[X^2]-(E[X])^2)

  sum = raw_squiggle_array[blockIdx.x * blockDim.x + threadIdx.x];
  sum2 = FMA_fp32(sum, sum, (0));
  // printf("1.sum=%0f\n", sum);
  // printf("1.sum2=%0f\n", sum2);

  sum = blockReduceSum(sum);
  sum2 = blockReduceSum(sum2);

  // printf("2.sum=%0f\n", sum);
  // printf("2.sum2=%0f\n", sum2);

  // printf("3.sum=%0f\n", sum);
  // printf("3.sum2=%0f\n", sum2);

  sum = DIV_fp32(sum, QUERY_LEN); // mean
  sum2 = DIV_fp32(sum2, QUERY_LEN);
  sum2 = SQRT_fp32(FMA_fp32(FMA_fp32(sum, sum, (0)), (-1),
                            sum2)); // stdev
  // printf("mean=%0f\n", sum);
  // printf("stdev=%0f\n", sum2);
  for (index_t i = blockDim.x * blockIdx.x + threadIdx.x;
       i < (gridDim.x * QUERY_LEN); i += blockDim.x) {
    raw_squiggle_array[i] = DIV_fp32(FMA_fp32(sum, -1, raw_squiggle_array[i]),
                                     sum2); // this is the normalized output
    // printf("raw_squiggle_array[%0d]=%0f\n", i, raw_squiggle_array[i]);
  }
}
#endif