#!/bin/bash

log1="summit/all_032619_1_2.log"
log2="summit/all_032619_2_2.log"

cd /ccs/home/lld/apps/miniVite

for(( j=0; j<3; j++ ))
do
  bsub -o $log1 submit_mid2.lsf
  bsub -o $log2 submit_hu.lsf
  sleep 1
  job_num=`bjob | grep lld | wc -l`
  while [ $job_num -ne 0 ]
  do
    sleep 30
    job_num=`bjob | grep lld | wc -l`
  done
done

cd -
