#!/bin/sh

if [ ! $# -eq 3 ]
then
    echo Usage: $0 stat_name print_index version
    exit
fi

./scripts/get_stats.sh "$1" $2 1 $3
echo ""
./scripts/get_stats.sh "$1" $2 2 $3
echo ""
./scripts/get_stats.sh "$1" $2 3 $3
echo ""
./scripts/get_stats.sh "$1" $2 4 $3
echo ""
./scripts/get_stats.sh "$1" $2 5 $3

#./scripts/get_all_stats.sh warmup 3 um | /ccs/home/lld/tools/v2m 15
#./scripts/get_all_stats.sh csr_ 3 um | /ccs/home/lld/tools/v2m 15
#./scripts/get_all_stats.sh total 4 um | /ccs/home/lld/tools/v2m 15
