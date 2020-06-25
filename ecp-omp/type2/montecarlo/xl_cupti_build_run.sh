#!/usr/bin/env bash

NVCC_EXE=`which nvcc`
NVCC_PATH=`dirname $NVCC_EXE`
CUDA_HOME=`dirname $NVCC_PATH`

XLC_OMP_FLAGS="-qoffload -fopenmp"

CUPTI_FLAGS="-I $CUDA_HOME/extras/CUPTI/include -L $CUDA_HOME/extras/CUPTI/lib64 -Wl,-rpath,$CUDA_HOME/extras/CUPTI/lib64 -lcupti -DCUDA_CUPTI=1 -DVERBOSE_MODE=1"

CC=xlc

CFLAGS="-g4 -O2"

$CC $CFLAGS $XLC_OMP_FLAGS $CUPTI_FLAGS bench_works.c -o bench_works_cupti

jsrun --smpiargs="-disable_gpu_hooks" --nrs 1 --tasks_per_rs 1 --cpu_per_rs 42 --gpu_per_rs 6 --rs_per_host 1 --latency_priority CPU-CPU --launch_distribution packed --bind rs ./bench_works_cupti 1000 500
