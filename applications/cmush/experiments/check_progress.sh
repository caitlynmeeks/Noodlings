#!/bin/bash
# Check dataset generation progress

echo "=== Dataset Generation Progress ==="
echo ""

# Check if process is running
if ps aux | grep -E "29199|generate_synthetic" | grep -v grep > /dev/null; then
    echo "Status: RUNNING"
else
    echo "Status: NOT RUNNING"
fi

echo ""

# Check output file sizes
for file in emotion_synthetic_*.json; do
    if [ -f "$file" ]; then
        size=$(wc -c < "$file")
        lines=$(wc -l < "$file")
        echo "$file: $size bytes, $lines lines"
    fi
done

echo ""

# Check log
if [ -f "generation_1000_proper_pad.log" ]; then
    log_size=$(wc -c < "generation_1000_proper_pad.log")
    echo "Log file: $log_size bytes"
    if [ $log_size -gt 0 ]; then
        echo "Last 10 lines:"
        tail -10 generation_1000_proper_pad.log
    fi
fi
