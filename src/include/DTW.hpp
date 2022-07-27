#ifndef DTW_HPP
#define DTW_HPP
#ifdef FP16
#include <cuda_fp16.h>
#endif
#include "datatypes.hpp"
namespace FullDTW {

#include "DTW.cu"

template <typename val_t, typename index_t>
__host__ void distances(reference_coefficients *ref, val_t *query, val_t *dists,
                        index_t num_entries, val_t thresh,
                        cudaStream_t stream) {

  DTW<index_t, val_t><<<num_entries, WARP_SIZE, 0, stream>>>(
      ref, query, dists, num_entries, thresh);

  return;
}

}; // namespace FullDTW

#endif
