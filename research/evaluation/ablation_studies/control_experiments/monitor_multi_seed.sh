#!/bin/bash

echo "========================================================================"
echo "MULTI-SEED REPLICATION - PROGRESS MONITOR"
echo "========================================================================"
echo

if ! pgrep -f "test_multi_seed.py" > /dev/null; then
    echo "❌ NOT RUNNING"
    echo
    if [ -f "multi_seed_results.json" ]; then
        echo "✅ COMPLETE - Results available in multi_seed_results.json"
    else
        echo "⚠️  No process found and no results file"
    fi
    exit 0
fi

echo "✅ RUNNING"
echo

# Extract progress from log
if [ -f "multi_seed_experiment.log" ]; then
    echo "Latest progress:"
    echo "----------------------------------------"

    # Find last "Replicating N=" line
    last_config=$(grep "Replicating N=" multi_seed_experiment.log | tail -1)
    if [ -n "$last_config" ]; then
        echo "$last_config"
    fi

    # Find last "Replication X/Y" line
    last_rep=$(grep "Replication" multi_seed_experiment.log | tail -1)
    if [ -n "$last_rep" ]; then
        echo "$last_rep"
    fi

    # Find last epoch line
    last_epoch=$(grep "Epoch" multi_seed_experiment.log | tail -1)
    if [ -n "$last_epoch" ]; then
        echo "$last_epoch"
    fi

    # Find last HSI line
    last_hsi=$(grep "HSI:" multi_seed_experiment.log | tail -3)
    if [ -n "$last_hsi" ]; then
        echo
        echo "Recent HSI values:"
        echo "$last_hsi"
    fi

    echo "----------------------------------------"
    echo

    # Calculate progress
    total_runs=30
    completed_runs=$(grep -c "^HSI:" multi_seed_experiment.log)
    echo "Completed runs: $completed_runs / $total_runs"

    if [ $completed_runs -gt 0 ]; then
        progress=$((completed_runs * 100 / total_runs))
        echo "Progress: $progress%"
    fi
else
    echo "No log file found yet..."
fi

echo
echo "========================================================================"
