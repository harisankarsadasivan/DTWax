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
#ifndef SCRAPPIE_STRUCTURES_H
#    define SCRAPPIE_STRUCTURES_H

#    include <inttypes.h>
#    include <stddef.h>

typedef struct {
    uint64_t start;
    float length;
    float mean;
    float stdv;
    int pos;
    int state;
} event_t;

typedef struct {
    size_t n;
    size_t start;
    size_t end;
    event_t *event;
} event_table;

typedef struct {
    size_t n;
    size_t start;
    size_t end;
    float *raw;
} raw_table;

#endif                          /* SCRAPPIE_DATA_H */
