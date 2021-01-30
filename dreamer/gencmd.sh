#!/bin/bash

# Takes in optional second argument which specifies the gpu
echo $1 | awk '{print $1"cmd.txt"}' | xargs cat | sed "s/python/srun --gpus ${2:-1} --qos kostas-med --partition kostas-compute --time 24:00:00 python3 -m pdb -c continue/"
