/*
// Not a contribution
// Changes made by NVIDIA CORPORATION & AFFILIATES enabling <XYZ> or otherwise
documented as
// NVIDIA-proprietary are not a contribution and subject to the following terms
and conditions:
 * SPDX-FileCopyrightText: Copyright (c) <year> NVIDIA CORPORATION & AFFILIATES.
All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 























































































































































































































































 # SPDX-FileCopyrightText: Copyright (c) <year> NVIDIA CORPORATION & AFFILIATES.
All rights reserved. # SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 #
 # NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 # property and proprietary rights in and to this material, related
 # documentation and any modifications thereto. Any use, reproduction,
 # disclosure or distribution of this material and related documentation
 # without an express license agreement from NVIDIA CORPORATION or
 # its affiliates is strictly prohibited.
 */
#ifndef DTW_HPP
#define DTW_HPP
#ifdef FP16
#include <cuda_fp16.h>
#endif
#include "datatypes.hpp"
namespace FullDTW {

#include "DTW.cu"

template <typename val_t, typename idx_t>

__host__ void distances(reference_coefficients *ref, val_t *query, val_t *dists,
                        idx_t num_entries, val_t thresh, cudaStream_t stream,
                        val_t *device_last_row) {

  DTW<idx_t, val_t><<<num_entries, WARP_SIZE, 0, stream>>>(
      ref, query, dists, num_entries, thresh, device_last_row);

  return;
}

}; // namespace FullDTW

#endif
