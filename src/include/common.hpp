#ifndef COMMON_HPP
#define COMON_HPP
#endif
//-------------global datatypes---------------------------//
typedef float value_t;    // data type for values
typedef uint64_t index_t; // data type for indices
typedef uint8_t label_t;  // data type for label
typedef int idxt;

#define SEGMENT_SIZE 64
#define WARP_SIZE 32
#define GROUP_SIZE WARP_SIZE
#define CELLS_PER_THREAD SEGMENT_SIZE
//...............global variables..........................//

#define QUERY_LEN 2048
#define BLOCK_NUM 1344
#define REF_LEN QUERY_LEN

/* calculate when to stop, and which thread has final result */
#define NUM_WAVES QUERY_LEN + (QUERY_LEN - 1) / CELLS_PER_THREAD
#define RESULT_THREAD_ID (QUERY_LEN - 1) / CELLS_PER_THREAD
#define RESULT_REG (QUERY_LEN - 1) % CELLS_PER_THREAD