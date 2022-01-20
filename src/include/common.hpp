#ifndef COMMON_HPP
#define COMON_HPP
#endif
//-------------global datatypes---------------------------//
typedef float value_t;    // data type for values
typedef uint64_t index_t; // data type for indices
typedef uint8_t label_t;  // data type for label
#define GROUP_SIZE 32
//...............global variables..........................//

#define QUERY_LEN 1024
#define BLOCK_NUM 1344
#define REF_LEN QUERY_LEN *BLOCK_NUM