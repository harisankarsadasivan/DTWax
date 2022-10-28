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
#All rights reserved. # SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 #
 # NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 # property and proprietary rights in and to this material, related
 # documentation and any modifications thereto. Any use, reproduction,
 # disclosure or distribution of this material and related documentation
 # without an express license agreement from NVIDIA CORPORATION or
 # its affiliates is strictly prohibited.
 */
#ifndef MAIN_PROG
#define MAIN_PROG

#include <assert.h>
#include <cstdint>
#include <iostream>

#include "include/common.hpp"
#include "include/datatypes.hpp"
#include <stdio.h>
#include <string>
#include <unistd.h>

#include "include/DTW.hpp"
#include "include/generate_load_squiggle.hpp"
#include "include/hpc_helpers.hpp"
#include "include/load_reference.hpp"
#include "include/normalizer.cu"
#include <unistd.h>

using namespace FullDTW;

//---------------------------------------------------------global
// vars----------------------------------------------------------//
cudaStream_t stream_var[STREAM_NUM];

int main(int argc, char **argv) {

  // create host storage and buffers on devices
  value_ht *host_query, // time series on CPU
      *host_dist,       // distance results on CPU
      // *host_ref_coeff1, *host_ref_coeff2,                // re-arranged ref
      // time series on CPU
      *device_query[STREAM_NUM], // time series on GPU
      *device_dist[STREAM_NUM];  // distance results on GPU

  value_ht *device_last_row[STREAM_NUM]; // stors last column of sub-matrix

  reference_coefficients *h_ref_coeffs, *d_ref_coeffs,
      *h_ref_coeffs_tmp; // struct stores reference genome's coeffs for DTW;
                         // *tmp is before restructuring for better mem
                         // coalescing
  raw_t *raw_array = NULL;
  std::vector<std::string> read_ids; // store read_ids to dump in output
  //****************************************************Target ref loading &
  // re-organization for better mem coalescing & target
  // loading****************************************//

  TIMERSTART(load_target)
  std::string ip_path = argv[1], model_file = argv[2], ref_file = argv[3];

  load_reference *REF_LD = new load_reference;

  REF_LD->ref_loader(ref_file);
  REF_LD->read_kmer_model(model_file);
  ASSERT(cudaMallocManaged(&h_ref_coeffs_tmp,
                           (sizeof(reference_coefficients) *
                            (REF_LEN)))); // host pinned memory for reference
  ASSERT(cudaMallocManaged(&h_ref_coeffs,
                           (sizeof(reference_coefficients) *
                            (REF_LEN)))); // host pinned memory for reference
  REF_LD->load_ref_coeffs(h_ref_coeffs_tmp);

  delete REF_LD;

  idxt k = 0;
  for (idxt l = 0; l < (REF_LEN / REF_TILE_SIZE); l += 1) {
    for (idxt i = 0; i < SEGMENT_SIZE; i++) {
      for (idxt j = 0; j < WARP_SIZE; j++) {
        h_ref_coeffs[k].coeff1 =
            h_ref_coeffs_tmp[(l * REF_TILE_SIZE) + (j * SEGMENT_SIZE) + i]
                .coeff1;
        // h_ref_coeffs[k].coeff2 =
        //     h_ref_coeffs_tmp[(l * REF_TILE_SIZE) + (j * SEGMENT_SIZE) + i]
        //         .coeff2;

        // std::cout << HALF2FLOAT(h_ref_coeffs[k].coeff1) << ","
        //           << HALF2FLOAT(h_ref_coeffs[k].coeff2) << "\n";
        k++;
      }
      // std::cout << "warp\n";
    }
  }

  cudaFree(h_ref_coeffs_tmp); // delete the tmp array

  ASSERT(
      cudaMalloc(&(d_ref_coeffs), (sizeof(reference_coefficients) * REF_LEN)));

  ASSERT(cudaMemcpyAsync(
      d_ref_coeffs,
      h_ref_coeffs, //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!//
      (sizeof(reference_coefficients) * REF_LEN), cudaMemcpyHostToDevice));

  TIMERSTOP(load_target)

  //*************************************************************LOAD FROM
  // FILE********************************************************//
  TIMERSTART(load_data)
  index_t NUM_READS; // counter to count number of reads to be
                     // processed + reference length
  squiggle_loader *loader = new squiggle_loader;
  loader->load_data(ip_path, raw_array, NUM_READS,
                    read_ids); // load from input ONT data folder with FAST5

  // NUM_READS = 1; //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!lkdnsknefkwnef
  ASSERT(cudaMallocHost(
      &raw_array,
      (sizeof(raw_t) *
       (NUM_READS * QUERY_LEN)))); // host pinned memory for raw data from FAST5

  loader->load_query(raw_array);

  delete loader;

  //****************************************************NORMALIZER****************************************//
  // normalizer instance - does h2h pinned mem transfer, CUDNN setup andzscore
  // normalization, normalized raw_t output is returned in same array as input
  normalizer *NMZR = new normalizer(NUM_READS);
  TIMERSTART(normalizer_kernel)

  NMZR->normalize(raw_array, NUM_READS, QUERY_LEN);

  TIMERSTOP(normalizer_kernel)
#ifdef NV_DEBUG
  std::cout << "cuDTW:: Normalizer processed  " << (QUERY_LEN * NUM_READS)
            << " raw samples in this time\n";
#endif

#ifdef NV_DEBUG
  NMZR->print_normalized_query(raw_array, NUM_READS, read_ids);
#endif

  delete NMZR;
  // normalizartion completed

  //****************************************************FLOAT to
  //__half2****************************************//
  ASSERT(cudaMallocHost(
      &host_query, sizeof(value_ht) * (NUM_READS * QUERY_LEN + WARP_SIZE))); /*
                     input */
  std::cout << "Normalized data:\n";

  for (index_t i = 0; i < NUM_READS; i++) {
    for (index_t j = 0; j < QUERY_LEN; j++) {
      host_query[(i * QUERY_LEN + j)] =
          FLOAT2HALF2(raw_array[(i * QUERY_LEN + j)]);
    }
  }
  for (index_t i = 0; i < WARP_SIZE; i++) {
    host_query[NUM_READS * QUERY_LEN + i] = FLOAT2HALF2(0.0f);
  }
  cudaFreeHost(raw_array);
  TIMERSTOP(load_data)

  //****************************************************MEM
  // allocation****************************************//
  TIMERSTART(malloc)
  //--------------------------------------------------------host mem
  // allocation--------------------------------------------------//

  // ASSERT(cudaMallocHost(&host_ref, sizeof(value_ht) * REF_LEN)); /* input

  ASSERT(cudaMallocHost(&host_dist, sizeof(value_ht) * NUM_READS)); /* results
                                                                     */

  //-------------------------------------------------------------dev mem
  // allocation-------------------------------------------------//

  for (int stream_id = 0; stream_id < STREAM_NUM; stream_id++) {
    ASSERT(
        cudaMalloc(&device_query[stream_id],
                   (sizeof(value_ht) * (BLOCK_NUM * QUERY_LEN + WARP_SIZE))));
    ASSERT(cudaMalloc(&device_dist[stream_id], (sizeof(value_ht) * BLOCK_NUM)));
    ASSERT(cudaStreamCreate(&stream_var[stream_id]));
    ASSERT(cudaMalloc(&device_last_row[stream_id],
                      (sizeof(value_ht) * (REF_LEN * BLOCK_NUM))));
  }

  TIMERSTOP(malloc)

  //****************************************************Mem I/O and DTW
  // computation****************************************//

  //-------------total batches of concurrent workload to & fro
  // device---------------//
  idxt batch_count = NUM_READS / (BLOCK_NUM * STREAM_NUM);
  std::cout << "Batch count: " << batch_count << " num_reads:" << NUM_READS
            << "\n";
  TIMERSTART_CUDA(concurrent_DTW_kernel_launch)
  for (idxt batch_id = 0; batch_id <= batch_count; batch_id += 1) {
    std::cout << "Processing batch_id: " << batch_id << "\n";

    idxt rds_in_batch = (BLOCK_NUM * STREAM_NUM);
    if (batch_id < batch_count)
      rds_in_batch = (BLOCK_NUM * STREAM_NUM);
    else if ((batch_id == batch_count) &&
             ((NUM_READS % (BLOCK_NUM * STREAM_NUM)) == 0)) {
      if (batch_count != 0)
        break;
      else
        rds_in_batch = NUM_READS;
    } else if ((batch_id == batch_count) &&
               ((NUM_READS % (BLOCK_NUM * STREAM_NUM)) != 0)) {
      rds_in_batch = NUM_READS % (BLOCK_NUM * STREAM_NUM);
    }
    for (idxt stream_id = 1; (stream_id <= STREAM_NUM) && (rds_in_batch != 0);
         stream_id += 1) {

      idxt rds_in_stream = BLOCK_NUM;

      if ((rds_in_batch - BLOCK_NUM) < 0) {
        rds_in_stream = rds_in_batch;
        rds_in_batch = 0;
      } else {
        rds_in_batch -= BLOCK_NUM;
        rds_in_stream = BLOCK_NUM;
      }
      std::cout << "Issuing " << rds_in_stream
                << " reads (blocks) from base addr:"
                << (batch_id * STREAM_NUM * BLOCK_NUM * QUERY_LEN) +
                       ((stream_id - 1) * BLOCK_NUM * QUERY_LEN)
                << " to stream_id " << (stream_id - 1) << "\n";
      //----h2d copy-------------//
      ASSERT(cudaMemcpyAsync(
          device_query[stream_id - 1],
          &host_query[(batch_id * STREAM_NUM * BLOCK_NUM * QUERY_LEN) +
                      ((stream_id - 1) * BLOCK_NUM * QUERY_LEN)],
          (sizeof(value_ht) * (QUERY_LEN * rds_in_stream + WARP_SIZE)),
          cudaMemcpyHostToDevice, stream_var[stream_id - 1]));

      //---------launch kernels------------//
      distances<value_ht, idxt>(d_ref_coeffs, device_query[stream_id - 1],
                                device_dist[stream_id - 1], rds_in_stream,
                                FLOAT2HALF2(0), stream_var[stream_id - 1],
                                device_last_row[stream_id - 1]);

      //-----d2h copy--------------//
      ASSERT(cudaMemcpyAsync(
          &host_dist[(batch_id * STREAM_NUM * BLOCK_NUM) +
                     ((stream_id - 1) * BLOCK_NUM)],
          device_dist[stream_id - 1], (sizeof(value_ht) * rds_in_stream),
          cudaMemcpyDeviceToHost, stream_var[stream_id - 1]));
    }
  }

  ASSERT(cudaDeviceSynchronize());
  TIMERSTOP_CUDA(concurrent_DTW_kernel_launch, NUM_READS)

  /* -----------------------------------------------------------------print
   * output -----------------------------------------------------*/

#ifndef FP16
  std::cout << "Read_ID\t"
            << "QUERY_LEN\t"
            << "REF_LEN\t"
            << "sDTW-score\n";
  for (index_t j = 0; j < NUM_READS; j++) {
    std::cout << j << "\t" << read_ids[j] << "\t" << QUERY_LEN << "\t"
              << REF_LEN << "\t" << HALF2FLOAT(host_dist[j]) << "\n";
  }
#else
  std::cout << "Read_ID\t"
            << "QUERY_LEN\t"
            << "REF_LEN\t"
            << "sDTW score: fwd-strand\tsDTW score: rev-strand\n";
  for (index_t j = 0; j < NUM_READS; j++) {
    std::cout << j << "\t" << read_ids[j] << "\t" << QUERY_LEN << "\t"
              << REF_LEN << "\t" << HALF2FLOAT(host_dist[j].x) << "\t"
              << HALF2FLOAT(host_dist[j].y) << "\n";
  }

#endif

  /* -----------------------------------------------------------------free
   * memory -----------------------------------------------------*/
  TIMERSTART(free)
  for (int stream_id = 0; stream_id < STREAM_NUM; stream_id++) {
    cudaFree(device_dist[stream_id]);
    cudaFree(device_query[stream_id]);
    cudaFree(device_last_row[stream_id]);
  }

  cudaFreeHost(host_query);
  cudaFreeHost(host_dist);
  cudaFree(h_ref_coeffs);
  cudaFree(d_ref_coeffs);

  TIMERSTOP(free)

  return 0;
}

#endif
