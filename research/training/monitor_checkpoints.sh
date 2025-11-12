#!/bin/bash
#
# Checkpoint Monitoring Script for Noodlings Training
#
# This script monitors training progress and notifies when checkpoints are ready
#

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

TRAINING_DIR="/Users/thistlequell/git/noodlings/training"
CHECKPOINTS_DIR="$TRAINING_DIR/checkpoints"
LOGS_DIR="$TRAINING_DIR/logs"

echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}  Noodlings Training Monitor${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""

# Check if training is running
TRAINING_PID=$(ps aux | grep -E "[p]ython.*train.*theory_of_mind|[p]ython.*train.*relationships|[p]ython.*train.*phase4" | awk '{print $2}' | head -1)

if [ -n "$TRAINING_PID" ]; then
    TRAINING_SCRIPT=$(ps aux | grep -E "[p]ython.*train" | head -1 | awk '{print $NF}')
    echo -e "${GREEN}✓ Training is RUNNING${NC}"
    echo -e "  PID: ${TRAINING_PID}"
    echo -e "  Script: ${TRAINING_SCRIPT}"
    echo ""
else
    echo -e "${YELLOW}⚠ No training process detected${NC}"
    echo ""
fi

# Find latest log file
LATEST_LOG=$(ls -t $LOGS_DIR/training_*.log 2>/dev/null | head -1)

if [ -n "$LATEST_LOG" ]; then
    echo -e "${BLUE}Latest Training Log:${NC}"
    echo -e "  ${LATEST_LOG}"
    echo ""

    # Show current epoch/progress
    CURRENT_EPOCH=$(tail -50 "$LATEST_LOG" | grep -E "Epoch [0-9]+/[0-9]+" | tail -1)
    if [ -n "$CURRENT_EPOCH" ]; then
        echo -e "${CYAN}Current Progress:${NC}"
        echo -e "  ${CURRENT_EPOCH}"
        echo ""
    fi

    # Check for errors
    ERROR_COUNT=$(grep -c "ERROR\|Error\|Traceback" "$LATEST_LOG" 2>/dev/null || echo "0")
    if [ "$ERROR_COUNT" -gt 0 ]; then
        echo -e "${RED}⚠ Found ${ERROR_COUNT} errors in log${NC}"
        echo -e "${YELLOW}Recent errors:${NC}"
        grep -A 2 "ERROR\|Error" "$LATEST_LOG" | tail -10
        echo ""
    fi
fi

# Check for checkpoints
echo -e "${BLUE}Available Checkpoints:${NC}"
echo ""

check_checkpoint_dir() {
    local dir=$1
    local name=$2

    if [ -d "$CHECKPOINTS_DIR/$dir" ]; then
        local checkpoint_count=$(find "$CHECKPOINTS_DIR/$dir" -name "*.npz" 2>/dev/null | wc -l | tr -d ' ')

        if [ "$checkpoint_count" -gt 0 ]; then
            echo -e "${GREEN}✓ ${name}:${NC} ${checkpoint_count} checkpoint(s) found"

            # Show latest checkpoint
            local latest=$(ls -t "$CHECKPOINTS_DIR/$dir"/*.npz 2>/dev/null | head -1)
            if [ -n "$latest" ]; then
                local size=$(du -h "$latest" | awk '{print $1}')
                local timestamp=$(stat -f "%Sm" -t "%Y-%m-%d %H:%M" "$latest" 2>/dev/null)
                echo -e "    Latest: $(basename $latest) (${size}, ${timestamp})"
            fi
        else
            echo -e "${YELLOW}○ ${name}:${NC} No checkpoints yet"
        fi
    else
        echo -e "${YELLOW}○ ${name}:${NC} Directory not created yet"
    fi
}

check_checkpoint_dir "theory_of_mind" "Theory of Mind"
check_checkpoint_dir "relationships" "Relationship Model"
check_checkpoint_dir "phase4_pretrain" "Phase 4 Full Model"
echo ""

# Estimate completion time
if [ -n "$TRAINING_PID" ] && [ -n "$LATEST_LOG" ]; then
    # Try to estimate from training speed
    RECENT_SPEED=$(tail -100 "$LATEST_LOG" | grep -Eo "[0-9]+\.[0-9]+it/s" | tail -1 | sed 's/it\/s//')

    if [ -n "$RECENT_SPEED" ]; then
        echo -e "${CYAN}Training Speed:${NC} ${RECENT_SPEED} examples/sec"

        # Rough estimate: 40K examples per epoch, 50 epochs for Theory of Mind
        # At ~450 it/s, that's about 90 seconds per epoch, ~75 minutes for 50 epochs
        echo -e "${CYAN}Estimated Time:${NC}"
        echo -e "  Theory of Mind: ~2-3 hours (50 epochs)"
        echo -e "  Relationships: ~1-2 hours (30 epochs)"
        echo -e "  Full Pipeline: ~3-5 hours total"
    fi
fi

echo ""
echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}To monitor in real-time, run:${NC}"
echo -e "  tail -f ${LOGS_DIR}/training_*.log"
echo -e "${CYAN}========================================${NC}"
