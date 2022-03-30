#ifndef CBF_GENERATOR_HPP
#define CBF_GENERATOR_HPP

#include "common.hpp"
#include <cstdint>
#include <random>

#ifndef FP16

void generate_cbf(value_t *data, index_t num_entries, index_t num_features,
                  uint64_t seed = 42) {

  std::default_random_engine generator(seed);
  std::uniform_int_distribution<int> distribution(0, 255);

#pragma omp parallel for

  for (index_t entry = 0; entry < num_entries * num_features; entry++) {

    data[entry] = distribution(generator);
    // #ifdef NV_DEBUG
    //     data[entry] = entry % 10;

    // #endif
  }
}

#else
template <typename raw_t>
void generate_cbf(raw_t *data, index_t num_entries, index_t num_features,
                  uint64_t seed = 42) {

  std::default_random_engine generator(seed);
  std::uniform_int_distribution<int> distribution(0, 255);

#pragma omp parallel for

  for (index_t entry = 0; entry < num_entries * num_features; entry++) {

    data[entry] = (raw_t)distribution(generator);
    // #ifdef NV_DEBUG
    //     data[entry] = entry % 10;

    // #endif
  }
}

#endif

#endif