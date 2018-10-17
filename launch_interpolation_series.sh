#!/bin/bash

factor=1000

for i in $(seq 0 30); do
    end=$(($factor * $i))
    if [ $i -eq 1 ]; then
	sbatch launch_interpolation.sh 0 $end
    else
	end=$(($factor * $i))
	sbatch launch_interpolation.sh $start $end
    fi
    start=$end
done
