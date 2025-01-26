#!/bin/bash

# get arguments for <num_simulations> <num_runs>
if [ $# -ne 2 ]; then
    echo "Usage: $0 <num_simulations> <num_runs>"
    exit 1
fi

# compile and run the program

#rm -f BSM_init

#g++ -O3 -march=armv8-a -mcpu=cortex-a57 BSM_init.cxx -o BSM_init

./BSMAD $1 $2