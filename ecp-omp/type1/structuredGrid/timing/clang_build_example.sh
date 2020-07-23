#!/usr/bin/env bash

NVCC_EXE=`which nvcc`
NVCC_PATH=`dirname $NVCC_EXE`
CUDA_HOME=`dirname $NVCC_PATH`

LLVM_PATH="$HOME/offload_llvm"
LLVM_OMP_FLAGS="-fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target -march=sm_70"
LLVM_LINK_FLAGS="-L$LLVM_PATH/lib -L$LLVM_PATH/lib64 -Wl,-rpath,$LLVM_PATH/lib -Wl,-rpath,$LLVM_PATH/lib64"

CUPTI_FLAGS="-I $CUDA_HOME/extras/CUPTI/include -L $CUDA_HOME/extras/CUPTI/lib64 -Wl,-rpath,$CUDA_HOME/extras/CUPTI/lib64 -lcupti -DCUDA_CUPTI=1"

CC=$LLVM_PATH/bin/clang

CFLAGS="-Wall -Wextra -O2 -march=native -g3 -fno-omit-frame-pointer"

$CC $CFLAGS $LLVM_OMP_FLAGS $LLVM_LINK_FLAGS $CUPTI_FLAGS example.c
