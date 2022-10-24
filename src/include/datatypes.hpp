#ifndef DATATYPES
#define DATATYPES
#include "common.hpp"
//-------------global datatypes---------------------------//
typedef float value_t;   // data type for values
typedef int64_t index_t; // data type for indices
typedef int8_t label_t;  // data type for label
typedef int idxt;
typedef float raw_t;

typedef struct ref_coefficients {
  value_ht coeff1; // coeff2;
} reference_coefficients;

#endif