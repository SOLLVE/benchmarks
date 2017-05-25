#!/bin/sh

if [ $# -eq 1 ]
then
    testv=$1
elif [ $# -eq 0 ]
then
    testv=um
    #testv=umpf
    #testv=h
    #testv=num
    #testv=hyb
else
    echo Usage: $0 [test_version]
    exit
fi

echo Test version is ${testv}

input=cant
inputpath=/ccs/home/lld/data/matrices/

./spmv_${testv} ${inputpath}${input}.mtx &> /dev/null
./spmv_${testv} ${inputpath}${input}.mtx
./spmv_${testv} ${inputpath}${input}.mtx
./spmv_${testv} ${inputpath}${input}.mtx
./spmv_${testv} ${inputpath}${input}.mtx
./spmv_${testv} ${inputpath}${input}.mtx

sleep 10
