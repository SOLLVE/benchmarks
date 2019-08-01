#!/bin/bash

log="summit/all_032619.log"
opt="-nnodes 1 -P GEN010SOLLVE -J km -W 120 -q batch -o $log"

cd /ccs/home/lld/apps/miniVite

for(( j=0; j<3; j++ ))
do
  for(( i=5000000; i<=150000000; i+=5000000 ))
  do
    LLD_GPU_MODE=UM bsub $opt jsrun -n1 -g6 nvprof ./miniVite -n 10000000
    sleep 1
    job_num=`bjob | grep lld | wc -l`
    while [ $job_num -ne 0 ]
    do
      sleep 30
      job_num=`bjob | grep lld | wc -l`
    done
  done
done

cd -
