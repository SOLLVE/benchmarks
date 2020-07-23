#!/usr/bin/env bash                                                                                                                                           

NVCC_EXE=`which nvcc`
NVCC_PATH=`dirname $NVCC_EXE`
CUDA_HOME=`dirname $NVCC_PATH`

XLC_OMP_FLAGS="-qoffload -fopenmp"

CUPTI_FLAGS="-I $CUDA_HOME/extras/CUPTI/include -L $CUDA_HOME/extras/CUPTI/lib64 -Wl,-rpath,$CUDA_HOME/extras/CUPTI/lib64 -lcupti -DCUDA_CUPTI=1 -DVERBOSE_MO\
DE=1"

CC=xlc

CFLAGS="-g4 -O2"

$CC $CFLAGS $XLC_OMP_FLAGS $CUPTI_FLAGS bench_works.c -o bench_works_cupti

$CC $CFLAGS $XLC_OMP_FLAGS $CUPTI_FLAGS bench_stencil.c -o bench_stencil_cupti


#jsrun --smpiargs="-disable_gpu_hooks" --nrs 1 --tasks_per_rs 1 --cpu_per_rs 42 --gpu_per_rs 6 --rs_per_host 1 --latency_priority CPU-CPU --launch_distributi\
on packed --bind rs ./bench_works_cupti 1000 500 1 10 > raw.log                                                                                               
#cat raw.log | grep KERNEL | sort | awk '{print $3,$4,$5}' > kernel_runtime.log                                                                               

#echo -n "KERNEL   time(us): "                                                                                                                                
#cat raw.log | grep KERNEL   | awk '{s+=$5} END {print s}'                                                                                                    
#echo -n "DRIVER   time(us): "                                                                                                                                
#cat raw.log | grep DRIVER   | awk '{s+=$3} END {print s}'                                                                                                    
#echo -n "MEMCPY   time(us): "                                                                                                                                
#cat raw.log | grep MEMCPY   | awk '{s+=$3} END {print s}'                                                                                                    
#echo -n "OVERHEAD time(us): "                                                                                                                                
#cat raw.log | grep OVERHEAD | awk '{s+=$3} END {print s}'                                                                                                    


# Put the numblocks(second parameter) as six to force task-to-GPU static schedule
jsrun --smpiargs="-disable_gpu_hooks" --nrs 1 --tasks_per_rs 1 --cpu_per_rs 42 --gpu_per_rs 6 --rs_per_host 1 --latency_priority CPU-CPU --launch_distributio\
n packed --bind rs ./bench_stencil_cupti 20000000 6 1 10 > raw.log
cat raw.log | grep KERNEL | sort | awk '{print $3,$4,$5}' > kernel_runtime_stencil.log

echo -n "KERNEL   time(us): "
cat raw.log | grep KERNEL   | awk '{s+=$5} END {print s}'
echo -n "DRIVER   time(us): "
cat raw.log | grep DRIVER   | awk '{s+=$3} END {print s}'
echo -n "MEMCPY   time(us): "
cat raw.log | grep MEMCPY   | awk '{s+=$3} END {print s}'
echo -n "OVERHEAD time(us): "
cat raw.log | grep OVERHEAD | awk '{s+=$3} END {print s}'

#jsrun --smpiargs="-disable_gpu_hooks" --nrs 1 --tasks_per_rs 1 --cpu_per_rs 42 --gpu_per_rs 6 --rs_per_host 1 --latency_priority CPU-CPU --launch_distributi\
on packed --bind rs ./bench_works_cupti 1000 500 1 1                                                                                                          




