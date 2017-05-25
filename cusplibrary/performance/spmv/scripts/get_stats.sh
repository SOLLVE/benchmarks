#!/bin/sh

stat="data size"
testv=um
input=cant
seqnum=1
idx=1

if [ $# -eq 4 ]
then
    stat=$1
    let idx=$2
    let seqnum=$3
    testv=$4
else
    echo Usage: $0 stat_name print_index sequence_number version
    exit
fi



#grep "${stat}" log/${input}-100-${testv}.log
#grep "${stat}" log/${input}-200-${testv}.log
#grep "${stat}" log/${input}-300-${testv}.log
#grep "${stat}" log/${input}-400-${testv}.log
#grep "${stat}" log/${input}-500-${testv}.log
#grep "${stat}" log/${input}-600-${testv}.log
#grep "${stat}" log/${input}-700-${testv}.log
#grep "${stat}" log/${input}-800-${testv}.log
#grep "${stat}" log/${input}-900-${testv}.log
#grep "${stat}" log/${input}-1000-${testv}.log
#grep "${stat}" log/${input}-1100-${testv}.log
#grep "${stat}" log/${input}-1200-${testv}.log
#grep "${stat}" log/${input}-1300-${testv}.log
#grep "${stat}" log/${input}-1400-${testv}.log
#grep "${stat}" log/${input}-1500-${testv}.log

grep "${stat}" log/${input}-100-${testv}.log    | awk "NR==${seqnum}" | awk "{print \$${idx}}"
grep "${stat}" log/${input}-200-${testv}.log    | awk "NR==${seqnum}" | awk "{print \$${idx}}"
grep "${stat}" log/${input}-300-${testv}.log    | awk "NR==${seqnum}" | awk "{print \$${idx}}"
grep "${stat}" log/${input}-400-${testv}.log    | awk "NR==${seqnum}" | awk "{print \$${idx}}"
grep "${stat}" log/${input}-500-${testv}.log    | awk "NR==${seqnum}" | awk "{print \$${idx}}"
grep "${stat}" log/${input}-600-${testv}.log    | awk "NR==${seqnum}" | awk "{print \$${idx}}"
grep "${stat}" log/${input}-700-${testv}.log    | awk "NR==${seqnum}" | awk "{print \$${idx}}"
grep "${stat}" log/${input}-800-${testv}.log    | awk "NR==${seqnum}" | awk "{print \$${idx}}"
grep "${stat}" log/${input}-900-${testv}.log    | awk "NR==${seqnum}" | awk "{print \$${idx}}"
grep "${stat}" log/${input}-1000-${testv}.log   | awk "NR==${seqnum}" | awk "{print \$${idx}}"
grep "${stat}" log/${input}-1100-${testv}.log   | awk "NR==${seqnum}" | awk "{print \$${idx}}"
grep "${stat}" log/${input}-1200-${testv}.log   | awk "NR==${seqnum}" | awk "{print \$${idx}}"
grep "${stat}" log/${input}-1300-${testv}.log   | awk "NR==${seqnum}" | awk "{print \$${idx}}"
grep "${stat}" log/${input}-1400-${testv}.log   | awk "NR==${seqnum}" | awk "{print \$${idx}}"
grep "${stat}" log/${input}-1500-${testv}.log   | awk "NR==${seqnum}" | awk "{print \$${idx}}"
