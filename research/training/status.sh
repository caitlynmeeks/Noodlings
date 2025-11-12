#!/bin/bash
#
# Check training status at a glance
#

PROJECT_DIR="/Users/thitlequell/git/consilience"
TRAINING_DIR="$PROJECT_DIR/training"
LOGS_DIR="$TRAINING_DIR/logs"
CHECKPOINTS_DIR="$TRAINING_DIR/checkpoints"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  Consilience Training Status${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Check if training is running
if pgrep -f "02_train_theory_of_mind\|03_train_relationships\|04_train_phase4_full" > /dev/null; then
    echo -e "${GREEN}✓ Training is RUNNING${NC}"

    # Show what's running
    if pgrep -f "02_train_theory_of_mind" > /dev/null; then
        echo "  Stage: Theory of Mind (Stage 2/4)"
    elif pgrep -f "03_train_relationships" > /dev/null; then
        echo "  Stage: Relationship Model (Stage 3/4)"
    elif pgrep -f "04_train_phase4_full" > /dev/null; then
        echo "  Stage: Full Phase 4 Training (Stage 4/4)"
    fi
else
    echo -e "${YELLOW}⚠ Training is NOT running${NC}"
fi

echo ""

# Check latest log
if [ -f "$LOGS_DIR/training.log" ]; then
    echo "Latest log entries:"
    echo -e "${YELLOW}$(tail -n 10 "$LOGS_DIR/training.log")${NC}"
else
    echo "No training logs found"
fi

echo ""

# Check checkpoints
echo "Checkpoints:"
for dir in theory_of_mind relationships phase4_pretrain; do
    if [ -d "$CHECKPOINTS_DIR/$dir" ]; then
        count=$(ls -1 "$CHECKPOINTS_DIR/$dir" 2>/dev/null | wc -l)
        if [ $count -gt 0 ]; then
            echo -e "  $dir: ${GREEN}$count files${NC}"

            # Show latest checkpoint
            latest=$(ls -t "$CHECKPOINTS_DIR/$dir"/*.npz 2>/dev/null | head -n 1)
            if [ -n "$latest" ]; then
                size=$(du -h "$latest" 2>/dev/null | cut -f1)
                echo "    Latest: $(basename "$latest") ($size)"
            fi
        else
            echo -e "  $dir: ${YELLOW}empty${NC}"
        fi
    else
        echo -e "  $dir: ${RED}not found${NC}"
    fi
done

echo ""

# Show data status
echo "Training Data:"
if [ -d "$TRAINING_DIR/data/synthetic" ]; then
    if [ -f "$TRAINING_DIR/data/synthetic/metadata.json" ]; then
        echo -e "  Synthetic data: ${GREEN}generated${NC}"
        # Try to show example count
        if command -v jq &> /dev/null; then
            total=$(jq -r '.num_examples' "$TRAINING_DIR/data/synthetic/metadata.json" 2>/dev/null)
            if [ -n "$total" ]; then
                echo "    Total examples: $total"
            fi
        fi
    else
        echo -e "  Synthetic data: ${YELLOW}incomplete${NC}"
    fi
else
    echo -e "  Synthetic data: ${RED}not generated${NC}"
    echo "    Run: python3 training/scripts/00_generate_synthetic_data.py"
fi

echo ""

# Show disk usage
echo "Disk Usage:"
if [ -d "$TRAINING_DIR" ]; then
    total_size=$(du -sh "$TRAINING_DIR" 2>/dev/null | cut -f1)
    echo "  Training directory: $total_size"

    if [ -d "$TRAINING_DIR/checkpoints" ]; then
        checkpoint_size=$(du -sh "$TRAINING_DIR/checkpoints" 2>/dev/null | cut -f1)
        echo "  Checkpoints: $checkpoint_size"
    fi

    if [ -d "$TRAINING_DIR/data" ]; then
        data_size=$(du -sh "$TRAINING_DIR/data" 2>/dev/null | cut -f1)
        echo "  Data: $data_size"
    fi
fi

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# Check tmux session
if command -v tmux &> /dev/null; then
    if tmux has-session -t training 2>/dev/null; then
        echo ""
        echo "tmux session 'training' is active"
        echo "  Attach: tmux attach -t training"
        echo "  Detach: Ctrl+B, then D"
    fi
fi
