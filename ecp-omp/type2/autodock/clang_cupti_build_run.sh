#!/usr/bin/env bash

NVCC_EXE=`which nvcc`
NVCC_PATH=`dirname $NVCC_EXE`
CUDA_HOME=`dirname $NVCC_PATH`

# Mostly for running on Seawulf's 8-K80 nodes, so sm_37 is used
LLVM_PATH="/sw/summit/llvm/11.0.0-rc1/11.0.0-rc1-0"
LLVM_OMP_FLAGS="-fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target -march=sm_37"
LLVM_LINK_FLAGS="-L$LLVM_PATH/lib -L$LLVM_PATH/lib64 -Wl,-rpath,$LLVM_PATH/lib -Wl,-rpath,$LLVM_PATH/lib64"

CUPTI_FLAGS="-I $CUDA_HOME/extras/CUPTI/include -L $CUDA_HOME/extras/CUPTI/lib64 -Wl,-rpath,$CUDA_HOME/extras/CUPTI/lib64 -lcupti -DCUDA_CUPTI=1 -DVERBOSE_MODE=1"

#CC=$LLVM_PATH/bin/clang
CXX=$LLVM_PATH/bin/clang++

CFLAGS="-Wall -Wextra -O2 -march=native -g3 -fno-omit-frame-pointer -lm"

#$CC $CFLAGS $LLVM_OMP_FLAGS $LLVM_LINK_FLAGS $CUPTI_FLAGS bench.cpp -o bench_works_cupti
$CXX $CFLAGS $LLVM_OMP_FLAGS $LLVM_LINK_FLAGS $CUPTI_FLAGS bench.cpp -o bench_works_cupti

./bench_works_cupti > raw.log
cat raw.log | grep KERNEL | sort | awk '{print $3,$4,$5}' > kernel_runtime.log

echo -n "KERNEL   time(us): "
cat raw.log | grep KERNEL   | awk '{s+=$5} END {print s}'
echo -n "DRIVER   time(us): "
cat raw.log | grep DRIVER   | awk '{s+=$3} END {print s}'
echo -n "MEMCPY   time(us): "
cat raw.log | grep MEMCPY   | awk '{s+=$3} END {print s}'
echo -n "OVERHEAD time(us): "
cat raw.log | grep OVERHEAD | awk '{s+=$3} END {print s}'
