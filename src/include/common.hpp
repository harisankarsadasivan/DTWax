#ifndef COMMON_HPP
#define COMON_HPP
#endif
//-------------global datatypes---------------------------//
typedef float value_t;    // data type for values
typedef uint64_t index_t; // data type for indices
typedef uint8_t label_t;  // data type for label
#define GROUP_SIZE 32
#define SEGMENT_SIZE 32
//...............global variables..........................//

#define QUERY_LEN 1024
#define BLOCK_NUM 1344
#define REF_LEN QUERY_LEN

/* calculate when to stop, and which thread has final result */
#define NUM_WAVES QUERY_LEN + (QUERY_LEN - 1) / CELLS_PER_THREAD
#define RESULT_THREAD_ID (QUERY_LEN - 1) / CELLS_PER_THREAD
#define RESULT_REG (QUERY_LEN - 1) % CELLS_PER_THREAD