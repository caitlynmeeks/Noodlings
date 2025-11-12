#!/bin/bash
echo "================================================================================"
echo "CRITICAL THRESHOLD EXPERIMENT - LIVE MONITOR"
echo "================================================================================"
echo ""

# Check if process is running
if pgrep -f "test_critical_threshold.py" > /dev/null; then
    echo "✅ Experiment is RUNNING"
    
    # Get process info
    ps aux | grep test_critical_threshold | grep -v grep | head -1
    echo ""
    
    # Check for log file
    if [ -f threshold_experiment.log ]; then
        echo "📊 Latest output from log:"
        echo "--------------------------------------------------------------------------------"
        tail -30 threshold_experiment.log
        echo "--------------------------------------------------------------------------------"
        echo ""
        echo "Lines in log: $(wc -l < threshold_experiment.log)"
    else
        echo "⏳ Log file not created yet (output may be buffered)"
    fi
    
    # Check for results file
    if [ -f threshold_results.json ]; then
        echo ""
        echo "📈 Partial results detected:"
        echo "$(jq -r '.[] | "\(.num_observers) observers: HSI = \(.hsi."slow/fast" // "pending")"' threshold_results.json 2>/dev/null || echo "Parsing...")"
    fi
else
    echo "❌ Experiment is NOT running"
    
    # Check for completion
    if [ -f threshold_results.json ]; then
        echo "✅ Experiment may be COMPLETE!"
        echo ""
        echo "Results:"
        cat threshold_results.json | head -20
    fi
fi

echo ""
echo "================================================================================"
echo "To monitor continuously: watch -n 10 ./monitor_threshold.sh"
echo "================================================================================"
