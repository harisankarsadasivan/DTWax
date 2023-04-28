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
#ifndef CBF_GENERATOR_HPP
#define CBF_GENERATOR_HPP

#include "common.hpp"
#include "hpc_helpers.hpp"
#include <cstdint>
#include <random>

#include "../../fast5/nanopolish_fast5_io.cpp"
#include "../../fast5/nanopolish_fast5_loader.cpp"
#include <dirent.h>
#include <fstream>
#include <math.h>
#include <stdio.h>
#include <sys/types.h>

#ifdef READ_UNTIL
#include <chrono>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#endif

class squiggle_loader {
public:
  void load_data(std::string fn, value_t *host_query, index_t &no_of_reads,
                 std::vector<std::string> &read_ids);
#ifndef READ_UNTIL
  void load_query(raw_t *raw_array);
#else
  void load_query(raw_t *raw_array, const std::string &PIPE_NAME,
                  const int &PIPE_SIZE, std::vector<std::string> &read_ids);
#endif

private:
  std::vector<raw_t>
      squiggle_vector; // squiggle data read/generated is stored here.
  index_t invalid_rds = 0, valid_rds = 0;
  void read_fast5_from_folder(std::string path,
                              std::vector<std::string> &file_queue);
};

#ifndef READ_UNTIL
void squiggle_loader::load_query(raw_t *raw_array) {
#ifdef NV_DEBUG
  std::cout << "Query loaded:\n";
#endif
  for (index_t i = 0; i < (valid_rds * QUERY_LEN); i++) {

    raw_array[i] = squiggle_vector[i];
#ifdef NV_DEBUG

    std::cout << raw_array[i] << ",";
#endif
  }
#ifdef NV_DEBUG
  std::cout << "\n=================\n";
#endif
}
#else

void squiggle_loader::load_query(raw_t *raw_array, const std::string &PIPE_NAME,
                                 const int &PIPE_SIZE,
                                 std::vector<std::string> &read_ids) {

  std::ifstream pipe_fd(PIPE_NAME, std::ios::binary);

  // Define mutex lock
  // td::mutex mutex;

  // while (true) {
  // Read from named pipe
  char buffer[PIPE_SIZE];
  // mutex.lock();
#ifdef NV_DEBUG
  std::cerr << "before reading from pipe\n";
#endif
  pipe_fd.read(buffer, PIPE_SIZE);
  // mutex.unlock();
#ifdef NV_DEBUG
  std::cerr << "after reading from pipe\n";
#endif
  // Process data from named pipe
  for (int i = 0; i < PIPE_SIZE; i++) {

    raw_array[i] = (raw_t)(unsigned char)buffer[i];
    read_ids.push_back(std::to_string(i));
#ifdef NV_DEBUG

    std::cout << i << ":\t" << raw_array[i] << ",";
#endif
  }
#ifdef NV_DEBUG
  std::cout << "\n=================\n";
#endif
  // }

  // Close named pipe
  // pipe_fd.close();

  return;
}
#endif
void squiggle_loader::read_fast5_from_folder(
    std::string path, std::vector<std::string> &file_queue) {

  std::cout << "cuDTW:: Loading ONT " << ONT_FILE_FORMAT << " reads from "
            << path << "\n";
  struct dirent *entry;
  DIR *dir = opendir(&path[0]);
  if (dir == NULL) {
    return;
  }

  std::string fname;

  while ((entry = readdir(dir)) != NULL) {

    fname = (entry->d_name);
    if (fname.substr(fname.rfind('.') + 1) == ONT_FILE_FORMAT) {
      file_queue.push_back(path + fname);
#ifdef NV_DEBUG
      std::cout << "filename is " << path + fname << "\n";
#endif
    }
  }

  closedir(dir);
}

void squiggle_loader::load_data(std::string fn,

                                raw_t *raw_array, index_t &no_of_reads,
                                std::vector<std::string> &read_ids) {

  std::vector<std::string> file_queue;
  std::ifstream f(fn);
  std::string line;

  // read in list of all ONT input files
  read_fast5_from_folder(fn, file_queue);

  // for every multi-fast5 file, read in all reads.

  for (size_t i = 0; i < file_queue.size(); ++i) {
    fast5_file f5_file = fast5_open(file_queue[i]);
    if (!fast5_is_open(f5_file)) {
      // data.is_valid = false;
      continue;
    }

    std::vector<std::string> reads = fast5_get_multi_read_groups(f5_file);
    // std::cerr << fast5_get_read_id_single_fast5(f5_file);
    // #pragma omp parallel for
    // check every read in ONT file
    // std::cout << "Query from fast5:\n";
    for (size_t j = 0; j < reads.size(); j++) {

      assert(reads[j].find("read_") == 0);

      Fast5Data data;
      data.rt.raw = NULL;
      data.rt.n = 0;

      std::string read_name = reads[j].substr(5);
      data.channel_params = fast5_get_channel_params(
          f5_file, read_name); // This has to be done prior to accessing raw
                               // data !!!Dont remove.
      data.rt = fast5_get_raw_samples(f5_file, read_name, data.channel_params);

      if (data.rt.n < (QUERY_LEN + ADAPTER_LEN)) {
        invalid_rds++;
        continue;
      }

      else {

        valid_rds++;
        read_ids.push_back(reads[j]);
      }
#ifdef NV_DEBUG
      std::cout << read_name << "\n";
#endif
      for (index_t itr = ADAPTER_LEN; itr < (QUERY_LEN + ADAPTER_LEN); itr++) {
        squiggle_vector.push_back((raw_t)data.rt.raw[itr]);
#ifdef NV_DEBUG
        std::cout << squiggle_vector.back() << ",";
#endif
      }
#ifdef NV_DEBUG

      std::cout << "\n=================\n";
#endif
    }

    fast5_close(f5_file);
  }
  std::cout << "cuDTW:: " << file_queue.size() << " ONT " << ONT_FILE_FORMAT
            << " files read\n"
            << "cuDTW:: Short/invalid reads exempted:: " << invalid_rds
            << std::endl;
  std::cout << "cuDTW:: Valid reads (NUM_READS) loaded :: " << valid_rds
            << std::endl;
  no_of_reads = valid_rds;
}

#endif