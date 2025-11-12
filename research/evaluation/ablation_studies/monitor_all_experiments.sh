#!/bin/bash
#
# Monitor all Musical Hypothesis experiments
#

echo "======================================================================"
echo "MUSICAL HYPOTHESIS EXPERIMENTS - STATUS"
echo "======================================================================"
echo ""
echo "Testing whether observer networks follow harmonic principles"
echo "analogous to musical intervals (period ≈ 12 discovery)"
echo ""
echo "======================================================================"
echo ""

# Check each experiment
declare -A experiments=(
    ["harmonic_ratios"]="test_harmonic_ratios.py:17:harmonic_ratios_results.json"
    ["phase_spacing"]="test_phase_spacing.py:20:phase_spacing_results.json"
    ["modulo_12"]="test_modulo_12.py:60:modulo_12_results.json"
)

for exp_name in harmonic_ratios phase_spacing modulo_12; do
    IFS=':' read -r script_name total_configs results_file <<< "${experiments[$exp_name]}"

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Experiment: ${exp_name}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # Check if running
    if pgrep -f "$script_name" > /dev/null; then
        echo "Status: ✓ RUNNING"

        # Try to count progress
        log_file="${exp_name}_experiment.log"
        if [ -f "$log_file" ]; then
            completed=$(grep -c "✓\|✗" "$log_file" 2>/dev/null || echo "0")
            if [ "$completed" -gt 0 ]; then
                percent=$((completed * 100 / total_configs))
                echo "Progress: $completed / $total_configs configurations ($percent%)"

                # ETA
                remaining=$((total_configs - completed))
                eta=$((remaining * 2))
                echo "Estimated time remaining: ~${eta} minutes"
            else
                echo "Progress: Just starting..."
            fi
        else
            echo "Progress: Initializing..."
        fi
    elif [ -f "$results_file" ]; then
        echo "Status: ✓ COMPLETE"
        echo "Results file: $results_file"
    else
        echo "Status: ✗ NOT STARTED"
    fi

    echo ""
done

echo "======================================================================"
echo "OVERALL STATUS"
echo "======================================================================"
echo ""

total_running=0
total_complete=0
total_notstarted=0

for exp_name in harmonic_ratios phase_spacing modulo_12; do
    IFS=':' read -r script_name total_configs results_file <<< "${experiments[$exp_name]}"

    if pgrep -f "$script_name" > /dev/null; then
        ((total_running++))
    elif [ -f "$results_file" ]; then
        ((total_complete++))
    else
        ((total_notstarted++))
    fi
done

echo "Running:      $total_running / 3"
echo "Complete:     $total_complete / 3"
echo "Not Started:  $total_notstarted / 3"
echo ""

if [ $total_complete -eq 3 ]; then
    echo "🎉 ALL EXPERIMENTS COMPLETE!"
    echo ""
    echo "Generate visualizations:"
    echo "  python3 plot_harmonic_ratios.py"
    echo "  python3 plot_phase_spacing.py"
    echo "  python3 plot_modulo_12.py"
elif [ $total_running -gt 0 ]; then
    echo "⏳ Experiments in progress..."
    echo ""
    echo "Check back later or run: ./monitor_all_experiments.sh"
else
    echo "Ready to start experiments"
fi

echo ""
echo "======================================================================"
