#ifndef CBF_GENERATOR_HPP
#define CBF_GENERATOR_HPP

#include "common.hpp"
#include <cstdint>
#include <random>

#ifdef FAST5 // read input squiggles from FAST5 file
#include "../../fast5/nanopolish_fast5_io.cpp"
#include "../../fast5/nanopolish_fast5_loader.cpp"
#include <dirent.h>
#include <math.h>
#include <stdio.h>
#include <sys/types.h>
#endif

#ifndef FP16

void generate_cbf(std::vector<raw_t> &data, index_t num_entries,
                  index_t num_features, uint64_t seed = 42) {

  std::default_random_engine generator(seed);
  std::uniform_int_distribution<int> distribution(0, 255);

  //#pragma omp parallel for

  for (index_t entry = 0; entry < num_entries * num_features; entry++) {

    data.push_back(distribution(generator));
    // #ifdef NV_DEBUG
    //     data[entry] = entry % 10;

    // #endif
  }
}

#else
template <typename raw_t>
void generate_cbf(std::vector<raw_t> &data, index_t num_entries,
                  index_t num_features, uint64_t seed = 42) {

  std::default_random_engine generator(seed);
  std::uniform_int_distribution<int> distribution(0, 255);

  //#pragma omp parallel for

  for (index_t entry = 0; entry < num_entries * num_features; entry++) {

    data.push_back((raw_t)distribution(generator));
    // #ifdef NV_DEBUG
    //     data[entry] = entry % 10;

    // #endif
  }
}

#endif

#ifdef FAST5

void read_fast5_from_folder(std::string path,
                            std::vector<std::string> &file_queue) {
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
      std::cout << "filename is " << path + fname << "\n";
    }
  }

  closedir(dir);
}

void load_from_fast5_folder(std::string fn, std::vector<raw_t> &squiggle_data,
                            index_t &no_of_reads) {

  std::vector<std::string> file_queue;
  std::ifstream f(fn);
  std::string line;
  static index_t invalid_rds = 0, valid_rds = 0;

  // read in list of all ONT input files
  read_fast5_from_folder(fn, file_queue);

  // for every multi-fast5 file, read in all reads.

  for (size_t i = 0; i < file_queue.size(); ++i) {
    fast5_file f5_file = fast5_open(file_queue[i]);
    if (!fast5_is_open(f5_file)) {
      continue;
    }

    std::vector<std::string> reads = fast5_get_multi_read_groups(f5_file);

    // #pragma omp parallel for
    // check every read in ONT file
    // std::cout << "Query from fast5:\n";
    for (size_t j = 0; j < reads.size(); j++) {

      assert(reads[j].find("read_") == 0);

      Fast5Data data;
      std::string read_name = reads[j].substr(5);
      data.rt = fast5_get_raw_samples(f5_file, read_name, data.channel_params);

      if (data.rt.n < (QUERY_LEN + ADAPTER_LEN)) {
        invalid_rds++;
        continue;
      } else if (isnan(data.rt.raw[ADAPTER_LEN])) { /// check if read is invalid
        invalid_rds++;
        continue;
      } else
        valid_rds++;
      data.is_valid = true;
      data.read_name = read_name;
      data.channel_params = fast5_get_channel_params(f5_file, read_name);

      for (index_t itr = ADAPTER_LEN; itr < (QUERY_LEN + ADAPTER_LEN); itr++) {
        squiggle_data.push_back((raw_t)data.rt.raw[itr]);
        // std::cout << squiggle_data.back() << ",";
      }
      // std::cout << "\n=================";
    }

    fast5_close(f5_file);
  }
  std::cout << file_queue.size() << " ONT " << ONT_FILE_FORMAT
            << " files read\n"
            << "Short/invalid NaN reads exempted:: " << invalid_rds
            << std::endl;
  std::cout << "Valid reads loaded:: " << valid_rds << std::endl;
  no_of_reads = valid_rds;
}
#endif

#endif