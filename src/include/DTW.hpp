#ifndef DTW_HPP
#define DTW_HPP

namespace FullDTW {

#include "DTW.cu"

template <typename value_t, typename index_t>
__host__ void distances(value_t *seqs, value_t *dists, index_t num_features,
                        index_t num_entries, value_t thresh) {

  for (index_t seq_idx = 0; seq_idx < num_entries; seq_idx++) {
    FullDTW<<<num_entries, 32, 0>>>(seqs, &seqs[seq_idx * num_features],
                                    &dists[seq_idx * num_entries], num_entries,
                                    num_features, thresh);
  }
  return;
}

}; // namespace FullDTW

#endif
