#!/bin/bash
# Array of room X=Y side length
SIDE=("50"  "60"  "70"  "80")
# Number of count in [1, MAXCOUNT]
MAXCOUNT=10
for ((i=8;i<=MAXCOUNT;i++)); do
    echo $i
    for l in ${SIDE[*]}; do
        echo $l
        python3 /home/zdai/repos/pyroomacoustics/pra_workspace/pra_creator.py --count $i --X $l --Y $l --samples 3
    done
done
