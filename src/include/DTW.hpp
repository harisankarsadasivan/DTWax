#ifndef DTW_HPP
#define DTW_HPP

namespace FullDTW {

#include "DTW.cu"

template <typename value_t, typename index_t>
__host__ void distances(value_t *ref, value_t *query, value_t *dists,
                        index_t num_entries, value_t thresh) {

  // for (index_t seq_idx = 0; seq_idx < num_entries; seq_idx++) {
  DTW<<<num_entries, WARP_SIZE, 0>>>(ref, query, dists, num_entries,
                                         thresh);
  //}
  return;
}

}; // namespace FullDTW

#endif
