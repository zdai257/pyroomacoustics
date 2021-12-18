#!/bin/bash
# Array of room X=Y side length
SIDE=("100")
# Number of count in [1, MAXCOUNT]
MAXCOUNT=10
for ((i=1;i<=MAXCOUNT;i++)); do
    echo $i
    for l in ${SIDE[*]}; do
        echo $l
        python3 /home/zdai/repos/pyroomacoustics/pra_workspace/pra_creator.py --count $i --X $l --Y $l --Z $l --samples 200
    done
done
