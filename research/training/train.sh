#!/bin/bash
#
# Consilience Phase 4 Master Training Script
#
# USAGE:
#   ./training/train.sh                    # Run full training pipeline
#   ./training/train.sh --resume           # Resume from checkpoint
#   ./training/train.sh --stage 2          # Start from specific stage
#
# This script is designed to be ADHD-friendly:
# - Resumable at any point
# - Clear progress indicators
# - Saves everything automatically
# - Tells you what's happening
#

set -e  # Exit on error

# Colors for pretty output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Directories
PROJECT_DIR="/Users/thistlequell/git/noodlings"
TRAINING_DIR="$PROJECT_DIR/training"
SCRIPTS_DIR="$TRAINING_DIR/scripts"
LOGS_DIR="$TRAINING_DIR/logs"
CHECKPOINTS_DIR="$TRAINING_DIR/checkpoints"
VENV_DIR="$PROJECT_DIR/venv"

cd $PROJECT_DIR

# Activate virtual environment
if [ ! -d "$VENV_DIR" ]; then
    log_error "Virtual environment not found at $VENV_DIR"
    log "Please run: python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

source "$VENV_DIR/bin/activate"

# Parse arguments
RESUME=false
START_STAGE=1

while [[ $# -gt 0 ]]; do
    case $1 in
        --resume)
            RESUME=true
            shift
            ;;
        --stage)
            START_STAGE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create log file
LOG_FILE="$LOGS_DIR/training_$(date +%Y%m%d_%H%M%S).log"
mkdir -p $LOGS_DIR

# Also create a symlink to latest log
ln -sf "$(basename $LOG_FILE)" "$LOGS_DIR/training.log"

# Logging function
log() {
    echo -e "${GREEN}[$(date +%H:%M:%S)]${NC} $1" | tee -a $LOG_FILE
}

log_stage() {
    echo "" | tee -a $LOG_FILE
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}" | tee -a $LOG_FILE
    echo -e "${BLUE}$1${NC}" | tee -a $LOG_FILE
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}" | tee -a $LOG_FILE
    echo "" | tee -a $LOG_FILE
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a $LOG_FILE
}

log_success() {
    echo -e "${GREEN}✓${NC} $1" | tee -a $LOG_FILE
}

# Banner
echo -e "${BLUE}"
cat << "EOF"
   ____                _ _ _
  / ___|___  _ __  ___(_) (_) ___ _ __   ___ ___
 | |   / _ \| '_ \/ __| | | |/ _ \ '_ \ / __/ _ \
 | |__| (_) | | | \__ \ | | |  __/ | | | (_|  __/
  \____\___/|_| |_|___/_|_|_|\___|_| |_|\___\___|

  Phase 4 Training Pipeline
  M3 Ultra 512GB Edition

EOF
echo -e "${NC}"

log "Training started"
log "Log file: $LOG_FILE"

# Check if resuming
if [ "$RESUME" = true ]; then
    log_stage "RESUMING TRAINING"
    # TODO: Load last stage from progress.json
    # For now, just continue
fi

# Stage 1: Data Generation (if needed)
if [ $START_STAGE -le 1 ]; then
    log_stage "STAGE 1: Data Generation"

    if [ -f "$TRAINING_DIR/data/synthetic/train.json" ]; then
        log "Synthetic data already exists, skipping generation"
    else
        log "Generating synthetic training data (50,000 examples)..."
        python3 $SCRIPTS_DIR/00_generate_synthetic_data.py 2>&1 | tee -a $LOG_FILE

        if [ ${PIPESTATUS[0]} -eq 0 ]; then
            log_success "Data generation complete"
        else
            log_error "Data generation failed"
            exit 1
        fi
    fi
fi

# Stage 2: Theory of Mind Pretraining
if [ $START_STAGE -le 2 ]; then
    log_stage "STAGE 2: Theory of Mind Pretraining (~2-3 hours)"

    log "Training Theory of Mind module on agent state inference..."
    python3 $SCRIPTS_DIR/02_train_theory_of_mind.py 2>&1 | tee -a $LOG_FILE

    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        log_success "Theory of Mind training complete"
    else
        log_error "Theory of Mind training failed"
        exit 1
    fi
fi

# Stage 3: Relationship Model Pretraining
if [ $START_STAGE -le 3 ]; then
    log_stage "STAGE 3: Relationship Model Pretraining (~1-2 hours)"

    log "Training Relationship Model on attachment/trust patterns..."
    python3 $SCRIPTS_DIR/03_train_relationships.py 2>&1 | tee -a $LOG_FILE

    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        log_success "Relationship Model training complete"
    else
        log_error "Relationship Model training failed"
        exit 1
    fi
fi

# Stage 4: Full Phase 4 End-to-End Training
if [ $START_STAGE -le 4 ]; then
    log_stage "STAGE 4: Phase 4 End-to-End Training (~24-48 hours)"

    log "Training full Phase 4 model with curriculum learning..."
    log "This is the long one - go touch grass, we'll be here when you get back"
    python3 $SCRIPTS_DIR/04_train_phase4_full.py 2>&1 | tee -a $LOG_FILE

    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        log_success "Phase 4 training complete!"
    else
        log_error "Phase 4 training failed"
        exit 1
    fi
fi

# Done!
log_stage "✓ TRAINING COMPLETE!"

log "All stages completed successfully"
log "Results:"
log "  - Theory of Mind checkpoint: $CHECKPOINTS_DIR/theory_of_mind/best.npz"
log "  - Relationship Model checkpoint: $CHECKPOINTS_DIR/relationships/best.npz"
log "  - Phase 4 pretrained model: $CHECKPOINTS_DIR/phase4_pretrain/best.npz"
log ""
log "Next steps:"
log "  1. Test model: python3 consilience_core/example_phase4_usage.py"
log "  2. Copy best checkpoint to release: cp $CHECKPOINTS_DIR/phase4_pretrain/best.npz models/consilience_phase4_pretrained.npz"
log "  3. Celebrate! 🎉"

# Send notification (if terminal-notifier installed)
if command -v terminal-notifier &> /dev/null; then
    terminal-notifier -title "Consilience Training" -message "Phase 4 training complete!" -sound default
fi

echo ""
echo -e "${GREEN}════════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}Training pipeline finished! Check the logs above for results.${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════════════════════${NC}"
