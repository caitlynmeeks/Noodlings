#!/bin/bash
#
# Automatically run topology experiment when oscillation mapping completes
#

echo "=============================================================================="
echo "TOPOLOGY EXPERIMENT LAUNCHER"
echo "=============================================================================="
echo ""
echo "Waiting for oscillation mapping to complete..."
echo ""

# Wait for oscillation mapping to finish
while pgrep -f "test_oscillation_mapping.py" > /dev/null; do
    echo -n "."
    sleep 60  # Check every minute
done

echo ""
echo ""
echo "✓ Oscillation mapping complete!"
echo ""
echo "Starting topology experiment in 10 seconds..."
sleep 10

echo ""
echo "=============================================================================="
echo "LAUNCHING TOPOLOGY ABLATION STUDY"
echo "=============================================================================="
echo ""

# Run topology experiment
python3 test_observer_topology.py 2>&1 | tee topology_experiment.log

echo ""
echo "=============================================================================="
echo "TOPOLOGY EXPERIMENT COMPLETE!"
echo "=============================================================================="
echo ""
echo "Generating visualizations..."
python3 plot_topology.py

echo ""
echo "✓ All experiments complete!"
echo "✓ Results in: results/"
echo "✓ Visualizations: topology_analysis.png"
echo ""
echo "🎉 Full ablation study chain complete!"
echo ""
