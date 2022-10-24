/*
 * SPDX-FileCopyrightText: Copyright (c) <year> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 
 # SPDX-FileCopyrightText: Copyright (c) <year> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 # SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 #
 # NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 # property and proprietary rights in and to this material, related
 # documentation and any modifications thereto. Any use, reproduction,
 # disclosure or distribution of this material and related documentation
 # without an express license agreement from NVIDIA CORPORATION or
 # its affiliates is strictly prohibited.
 */
#pragma once
#ifndef SCRAPPIE_COMMON_H
#define SCRAPPIE_COMMON_H

#include "scrappie_structures.h"

raw_table trim_and_segment_raw(raw_table rt, int trim_start, int trim_end, int varseg_chunk, float varseg_thresh);
raw_table trim_raw_by_mad(raw_table rt, int chunk_size, float proportion);

#endif /* SCRAPPIE_COMMON_H */
