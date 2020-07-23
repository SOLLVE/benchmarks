!/usr/bin/env bash                                                                                                                                           

NVCC_EXE=`which nvcc`
NVCC_PATH=`dirname $NVCC_EXE`
CUDA_HOME=`dirname $NVCC_PATH`

# Mostly for running on Seawulf's 8-K80 nodes, so sm_37 is used                                                                                               
LLVM_PATH="$HOME/offload_llvm"
LLVM_OMP_FLAGS="-fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target -march=sm_37"
LLVM_LINK_FLAGS="-L$LLVM_PATH/lib -L$LLVM_PATH/lib64 -Wl,-rpath,$LLVM_PATH/lib -Wl,-rpath,$LLVM_PATH/lib64"

CUPTI_FLAGS="-I $CUDA_HOME/extras/CUPTI/include -L $CUDA_HOME/extras/CUPTI/lib64 -Wl,-rpath,$CUDA_HOME/extras/CUPTI/lib64 -lcupti -DCUDA_CUPTI=1 -DVERBOSE_MO\
DE=1"

CC=$LLVM_PATH/bin/clang

CFLAGS="-Wall -Wextra -O2 -march=native -g3 -fno-omit-frame-pointer -lm"

$CC $CFLAGS $LLVM_OMP_FLAGS $LLVM_LINK_FLAGS $CUPTI_FLAGS bench_stencil.c -o bench_stencil_cupti

jsrun --smpiargs="-disable_gpu_hooks" --nrs 1 --tasks_per_rs 1 --cpu_per_rs 42 --gpu_per_rs 6 --rs_per_host 1 --latency_priority CPU-CPU --launch_distribution packed --bind rs ./bench_stencil_cupti 1000 500 1 10 > raw.log
cat raw.log | grep KERNEL | sort | awk '{print $3,$4,$5}' > kernel_runtime.log

echo -n "KERNEL   time(us): "
cat raw.log | grep KERNEL   | awk '{s+=$5} END {print s}'
echo -n "DRIVER   time(us): "
cat raw.log | grep DRIVER   | awk '{s+=$3} END {print s}'
echo -n "MEMCPY   time(us): "
cat raw.log | grep MEMCPY   | awk '{s+=$3} END {print s}'
echo -n "OVERHEAD time(us): "
cat raw.log | grep OVERHEAD | awk '{s+=$3} END {print s}'








