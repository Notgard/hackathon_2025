#!/bin/bash

# Directory for log files
LOG_DIR="bsm_logs_BSMAD"

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Test cases
declare -a TESTS=(
    "100000 10"
    "100000 100"
    "1000000 10"
    "1000000 100"
    "10000000 10"
    "10000000 100"
    "100000000 10"
    "100000000 100"
)

# Run all tests
for i in "${!TESTS[@]}"; do
    TEST="${TESTS[i]}"
    LOG_FILE="$LOG_DIR/test_$((i + 1)).log"
    PARAM_LOG_FILE="$LOG_DIR/test_$((i + 1))_avg.log"

    echo "Running ./BSM $TEST for 10 iterations..."
    total_time=0

    # Run each test 10 times
    for run in {1..10}; do
        RUN_LOG="$LOG_DIR/test_$((i + 1))_run_$run.log"
        ./run.sh $TEST > "$RUN_LOG" 2>&1

        # Extract the runtime from the output
        runtime=$(grep -oP 'Performance in seconds : \K[\d\.]+' "$RUN_LOG")
        total_time=$(echo "$total_time + $runtime" | bc)

        # Log individual run result
        echo "Run #$run: $runtime seconds" >> "$LOG_FILE"
    done

    # Calculate average runtime
    avg_time=$(echo "scale=6; $total_time / 10" | bc)

    # Log the average runtime
    echo "Parameter: $TEST" >> "$PARAM_LOG_FILE"
    echo "Average Runtime: $avg_time seconds" >> "$PARAM_LOG_FILE"
    echo "---------------------------" >> "$PARAM_LOG_FILE"

    echo "Results for ./BSM $TEST saved in $PARAM_LOG_FILE"
done

echo "Execution completed. Logs are in the $LOG_DIR directory."
