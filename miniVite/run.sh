#!/bin/bash

log0="summit/alloc_051919_lru_sm.log"
log1="summit/alloc_051919_lru_la.log"

cd /ccs/home/lld/apps/miniVite

for(( j=0; j<3; j++ ))
do
  bsub -o $log0 submit_sm.lsf
  sleep 1
  job_num=`bjob | grep lld | grep mnV | wc -l`
  while [ $job_num -ne 0 ]
  do
    sleep 20
    job_num=`bjob | grep lld | grep mnV | wc -l`
  done
  for(( i=50000000; i<=150000000; i+=10000000 ))
  do
    sed "s/input/$i/" < submit_one.lsf > temp.lsf
    bsub -o $log1 temp.lsf
    sleep 1
    job_num=`bjob | grep lld | grep mnV | wc -l`
    while [ $job_num -ne 0 ]
    do
      sleep 20
      job_num=`bjob | grep lld | grep mnV | wc -l`
    done
  done
done

cd -
