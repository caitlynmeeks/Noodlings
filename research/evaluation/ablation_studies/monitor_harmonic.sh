#!/bin/bash
#
# Monitor harmonic ratio experiment progress
#

echo "======================================================================"
echo "HARMONIC RATIO EXPERIMENT MONITOR"
echo "======================================================================"
echo ""

# Check if experiment is running
if pgrep -f "test_harmonic_ratios.py" > /dev/null; then
    echo "✓ Experiment is RUNNING"
    echo ""

    # Show last 30 lines of log
    if [ -f harmonic_ratios_experiment.log ]; then
        echo "Recent output:"
        echo "----------------------------------------------------------------------"
        tail -30 harmonic_ratios_experiment.log
        echo "----------------------------------------------------------------------"
        echo ""

        # Count completed configurations
        completed=$(grep -c "✓ STABLE\|✗ UNSTABLE" harmonic_ratios_experiment.log 2>/dev/null || echo "0")
        # Remove any whitespace/newlines
        completed=$(echo "$completed" | tr -d '\n\r ')
        echo "Progress: $completed / 17 configurations completed"

        # Estimate remaining time (assuming ~2 min per config)
        if [ "$completed" -gt 0 ] 2>/dev/null; then
            remaining=$((17 - completed))
            eta=$((remaining * 2))
            echo "Estimated time remaining: ~${eta} minutes"
        else
            echo "Just starting..."
        fi
    else
        echo "Log file not created yet..."
    fi
else
    echo "✗ Experiment is NOT running"
    echo ""

    # Check if results exist
    if [ -f harmonic_ratios_results.json ]; then
        echo "✓ Results file found!"
        echo ""
        echo "Run: python3 plot_harmonic_ratios.py"
        echo "  to generate visualizations"
    else
        echo "No results file found yet."
    fi
fi

echo ""
echo "======================================================================"
