#!/bin/bash
### Begin BSUB Options
#BSUB -P STF010
#BSUB -J MULTI_GPU_OMP
#BSUB -W 00:01
#BSUB -nnodes 1
#BSUB -alloc_flags "smt4"
### End BSUB Options and begin shell commands

export OMP_NUM_THREADS=42
cd /gpfs/alpine/scratch/oscarh/stf010/oscar-bench
jsrun --nrs 1 --tasks_per_rs 1 --cpu_per_rs 42 --gpu_per_rs 6 --rs_per_host 1 --latency_priority CPU-CPU --launch_distribution packed --bind rs ./bench