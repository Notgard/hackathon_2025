#!/bin/bash

# Define the path to the Code_Aster executable
CODE_ASTER_BIN="./run_aster"

# Define the input file for the benchmarks
INPUT_FILE="Cube_perf.py"

# Define the REFINE values to test
REFINE_VALUES=(3 6 7 8 9)

# Loop over each REFINE value
for REFINE in "${REFINE_VALUES[@]}"; do
    # Define the output log file for this run
    LOG_FILE="benchmark_refine_${REFINE}.log"

    # Print a message indicating the start of the benchmark
    echo "Running benchmark with REFINE=${REFINE}..."

    # Run the benchmark with the current REFINE value and log the output
    REFINE=$REFINE $CODE_ASTER_BIN --memory_limit=15000 $INPUT_FILE > $LOG_FILE 2>&1

    # Print a message indicating the end of the benchmark
    echo "Benchmark with REFINE=${REFINE} completed. Results saved to ${LOG_FILE}."
done

echo "All benchmarks completed."
