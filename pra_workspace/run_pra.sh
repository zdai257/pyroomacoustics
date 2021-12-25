#!/bin/bash
# Array of room X=Y side length
SIDE=("100")
# Number of count in [1, MAXCOUNT]
SNR=("-20"  "-33")
# Number of count in [1, MAXCOUNT]
MAXCOUNT=5
for ((i=1;i<=MAXCOUNT;i++)); do
    echo $i
    for l in ${SIDE[*]}; do
        echo $l
        for s in ${SNR[*]}; do
            python3 /home/zdai/repos/pyroomacoustics/pra_workspace/pra_creator.py --count $i --X $l --Y $l --Z $l --samples 2000 --snr $s --rt_order 2 --absorb 0.8
        done
    done
done
