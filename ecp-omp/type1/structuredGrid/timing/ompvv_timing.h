#ifndef __OMPVV_TIMING__
#define __OMPVV_TIMING__

#include "ompvv.h"
#include <math.h>
#include <sys/time.h>
#include <stdint.h>


#ifndef CUDA_CUPTI
#ifndef NUM_REP
#define NUM_REP 3
#endif
#define OMPVV_GET_TIME(timer) \
{ \
  struct timeval time;\
  gettimeofday(&time, NULL);\
  timer = (uint64_t)time.tv_sec*1e6 + (uint64_t)time.tv_usec; \
}

#define OMPVV_PRINT_TIME_LAPSED(start, stop) \
{ \
  OMPVV_INFOMSG("Time(us) = %ju", stop - start)\
}


#define OMPVV_INIT_TIMERS

#define OMPVV_TIMING_LOAD

#define OMPVV_START_TIMER
#define OMPVV_STOP_TIMER
#define OMPVV_GET_TIME_LAPSED

#define OMPVV_INIT_TEST

// Implementing insertion sort 
#define OMPVV_REGISTER_TEST

// Find average, median and standard deviation ignore first and last (list of results is sorted)
#define OMPVV_TIMER_RESULT(clause)

#define OMPVV_PRINT_VALUES


#else // CUDA PROFILER VERSION

#define NUM_REP 1

#include <stdio.h>
#include <cuda.h>
#include <cupti.h>
#include "ompvv_cupti.h"

#define OMPVV_INIT_TIMERS  \
initTrace();

#define OMPVV_PRINT_TIME_LAPSED(start, stop) 

#define OMPVV_TIMING_LOAD \
{ \
  uint64_t volatile __ompvv_b = 0;\
    __ompvv_b++; \
}

#define OMPVV_START_TIMER \
  cuptiActivityFlushAll(0); \
  ignoreEventsPrinting = 0; \
  _ompvv_accum_driver = 0; \
  _ompvv_accum_kernel = 0; \
  _ompvv_accum_runtime = 0; \
  _ompvv_accum_memory = 0; \
  _ompvv_accum_others = 0;

#define OMPVV_STOP_TIMER  \
      cuptiActivityFlushAll(0);

#define OMPVV_INIT_TEST \
  OMPVV_INFOMSG("Starting test");

#define OMPVV_REGISTER_TEST \
  ignoreEventsPrinting = 1;

#define OMPVV_TIMER_RESULT(clause) \
  do { /* All useless, removed */ } while (0);

#endif

#endif
