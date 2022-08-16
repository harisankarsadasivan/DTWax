#ifndef LOAD_REFERENCE
#define LOAD_REFERENCE

#include "common.hpp"
#include "datatypes.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
// #include <iomanip>
#include <iostream>
#include <map>
#include <string>

class load_reference {
public:
  raw_t *ref_mean, ref_stdev;
  // std::map<int, std::tuple<raw_t, raw_t>> kmer_model;
  void ref_loader(std::string fname);
  void read_kmer_model(std::string fname);
  value_ht *ref_coeff1 = NULL;
  // ref_coeff2 = NULL; // coeff1 is 1/stdev and coeff2 is mean/stdev
  void load_ref_coeffs(reference_coefficients *ref);

private:
  std::string fwd_reference, rev_reference,
      reference; // reference is concatenation of fwd and reverse
  std::map<std::string, std::tuple<raw_t, raw_t>>
      kmer_model; // maps 1/stdev and mean/stdev to kmer
  void complement(std::string &DNAseq);
};

// load reference genome's coefficients and ref length
void load_reference::load_ref_coeffs(reference_coefficients *ref) {
  std::string target[2] = {fwd_reference, rev_reference};
#ifndef FP16
  idxt start_idx[2] = {0, REF_LEN / 2};
  index_t ref_len = REF_LEN / 2;
#else
  idxt start_idx[2] = {0, 0};
  index_t ref_len = REF_LEN;
#endif
  for (idxt j = 0; j < 2; j++) {
    raw_t mean1 = 0, stdev1 = 0; // mean2 = 0, stdev2 = 0;

    // calculates mean and stdev
    for (index_t i = 0; i < ref_len; i++) {
      // std::cout << fwd_reference.substr(i, KMER_LEN) << ",";
      for (std::map<std::string, std::tuple<raw_t, raw_t>>::iterator itr =
               kmer_model.begin();
           itr != kmer_model.end(); ++itr) {
        if (target[j].substr(i, KMER_LEN) == itr->first) {
          mean1 += std::get<0>(itr->second);
          // mean2 += std::get<1>(itr->second);
          stdev1 =
              stdev1 + (std::get<0>(itr->second)) * (std::get<0>(itr->second));
          // stdev2 =
          //    stdev2 + (std::get<1>(itr->second)) *
          //    (std::get<1>(itr->second));
          break;
        }
      }
    }

    mean1 = mean1 / ref_len;
    // mean2 = mean2 / ref_len;
    stdev1 = stdev1 / ref_len;
    // stdev2 = stdev2 / ref_len;
    stdev1 = sqrt(stdev1 - (mean1 * mean1));
    // stdev2 = sqrt(stdev2 - (mean2 * mean2));

#ifdef NV_DEBUG
    std::cout << "Printing mean and stdev of time series before normalizing "
                 "squiggle ref coefficients: "
              << mean1 << ", " << stdev1 << "\n";
#endif
    float coeff1; // coeff2;
    // z-score normalize the reference coefficients
    for (index_t i = 0; i < ref_len; i++) {
      // std::cout << fwd_reference.substr(i, KMER_LEN) << ",";
      for (std::map<std::string, std::tuple<raw_t, raw_t>>::iterator itr =
               kmer_model.begin();
           itr != kmer_model.end(); ++itr) {
        if (target[j].substr(i, KMER_LEN) == itr->first) {
          coeff1 = (std::get<0>(itr->second) - mean1) / stdev1;
          // coeff2 = (std::get<1>(itr->second) - mean2) / stdev2;
#ifdef NV_DEBUG
          std::cout << "[ " << coeff1 << "], ";
          // std::cout << coeff1 << ", " << coeff2 << "\n";
#endif

#ifndef FP16
          ref[start_idx[j] + i].coeff1 = FLOAT2HALF(coeff1);
#else
          if (j == 0)
            ref[start_idx[j] + i].coeff1.x = __float2half_rn(coeff1);
          else
            ref[start_idx[j] + i].coeff1.y = __float2half_rn(coeff1);
#endif
          // ref[start_idx[j] + i].coeff2 = FLOAT2HALF(1 / (2 * coeff2 *
          // coeff2));
        }
      }
    }
#ifdef NV_DEBUG
    std::cout << "\nPrinting end\n";
#endif
  }
}

// reverse complement base
void load_reference::complement(std::string &DNAseq) {
  for (std::size_t i = 0; i < DNAseq.length(); ++i) {
    switch (DNAseq[i]) {
    case 'A':
      DNAseq[i] = 'T';
      break;
    case 'C':
      DNAseq[i] = 'G';
      break;
    case 'G':
      DNAseq[i] = 'C';
      break;
    case 'T':
      DNAseq[i] = 'A';
      break;
    }
  }
}

// read kmer squiggle model from .txt file for r9.4 chemistry
void load_reference::read_kmer_model(std::string fname) {
  std::ifstream f(fname);
  std::string line;
  while (std::getline(f, line)) {
    std::string kmer;
    raw_t curr_mean, curr_stdev;

    std::istringstream ss(line);

    ss >> kmer >> curr_mean >> curr_stdev;
    // std::cout << std::setprecision(8) << curr_mean << " ,";
    kmer_model.insert(
        std::make_pair(kmer, std::make_tuple((curr_mean), (curr_stdev))));
    // std::make_tuple((1 / curr_stdev), (-1.0f * curr_mean / curr_stdev))));
  }
  // for
  // debugging:
  //   prints kmer to mean and stdev
  //       map
#ifdef NV_DEBUG
  std::cout << "kmer\t"
            << "mean\t"
            << "variance\n";
  for (std::map<std::string, std::tuple<raw_t, raw_t>>::iterator itr =
           kmer_model.begin();
       itr != kmer_model.end(); ++itr) {
    std::cout << itr->first << '\t' << std::get<0>(itr->second) << '\t'
              << std::get<1>(itr->second) << '\n';
  }
#endif
}

// loads a single entry basecalled reference from FASTA file
void load_reference::ref_loader(std::string fname) {

  std::ifstream input(fname);
  if (!input.good()) {
    std::cerr << "Error opening '" << fname << "'. Bailing out." << std::endl;
    // return -1;
  }

  std::string line, name, content;
  while (std::getline(input, line)) {
    if (line.empty() || line[0] == '>') { // Identifier marker
      if (!name.empty()) { // Print out what we read from the last entry
        std::cout << name << " : " << content << std::endl;
        name.clear();
      }
      if (!line.empty()) {
        name = line.substr(1);
      }
      content.clear();
    } else if (!name.empty()) {
      if (line.find(' ') !=
          std::string::npos) { // Invalid sequence--no spaces allowed
        name.clear();
        content.clear();
      } else {
        content += line;
      }
    }
  }
  if (!name.empty()) { // Print out what we read from the last entry
    fwd_reference = content;

    // reverse and complement base
    rev_reference = content;
    std::reverse(rev_reference.begin(), rev_reference.end());
    load_reference::complement(rev_reference);

    // std::cout << name << " : " << content << std::endl;
    // std::cout << rev_reference << std::endl;
  } else {
    std::cerr << "Invalid FASTA entry in '" << fname << "'. Bailing out."
              << std::endl;
  }
  // fwd_reference = fwd_reference.substr(0, REF_LEN / 2);
  // rev_reference = rev_reference.substr(0, REF_LEN / 2);

  reference = fwd_reference + rev_reference; // concatenated reference
  // std::cout << fwd_reference.length() << "\n";
}

#endif