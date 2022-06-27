#ifndef LOAD_REFERENCE
#define LOAD_REFERENCE

#include "common.hpp"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <string>

class load_reference {
public:
  raw_t *ref_mean, ref_stdev;
  // std::map<int, std::tuple<raw_t, raw_t>> kmer_model;
  void ref_loader(std::string fname);
  void read_kmer_model(std::string fname);
  value_ht *ref_coeff1 = NULL,
           ref_coeff2 = NULL; // coeff1 is 1/stdev and coeff2 is mean/stdev
  void load_ref_coeffs(value_ht *ref_coeff1, value_ht *ref_coeff2);

private:
  std::string fwd_reference, rev_reference;
  std::map<std::string, std::tuple<raw_t, raw_t>>
      kmer_model; // maps 1/stdev and mean/stdev to kmer
  void complement(std::string &DNAseq);
};

// load reference genome's coefficients and ref length
void load_reference::load_ref_coeffs(value_ht *ref_coeff1,
                                     value_ht *ref_coeff2) {
  for (index_t i = 0; i < fwd_reference.length() - KMER_LEN + 1; i++) {
    // std::cout << fwd_reference.substr(i, KMER_LEN) << ",";
    for (std::map<std::string, std::tuple<raw_t, raw_t>>::iterator itr =
             kmer_model.begin();
         itr != kmer_model.end(); ++itr) {
      if (fwd_reference.substr(i, KMER_LEN) == itr->first) {
        ref_coeff1[i] = FLOAT2HALF(std::get<0>(itr->second));
        ref_coeff2[i] = FLOAT2HALF(std::get<1>(itr->second));
      }
    }
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
    kmer_model.insert(std::make_pair(
        kmer, std::make_tuple((1 / curr_stdev), (curr_mean / curr_stdev))));
  }
  // for
  // debugging:
  //   prints kmer to mean and stdev
  //       map
  // std::cout << "kmer\t"
  //           << "1/stdev\t"
  //           << "mean/stdev\n";
  // for (std::map<std::string, std::tuple<raw_t, raw_t>>::iterator itr =
  //          kmer_model.begin();
  //      itr != kmer_model.end(); ++itr) {
  //   std::cout << itr->first << '\t' << std::get<0>(itr->second) << '\t'
  //             << std::get<1>(itr->second) << '\n';
  // }
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
  fwd_reference = fwd_reference.substr(0, 1024);
  rev_reference = rev_reference.substr(0, 1024);
  // std::cout << fwd_reference.length() << "\n";
}

#endif