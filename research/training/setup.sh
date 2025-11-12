#!/bin/bash
#
# Training Setup Script
#
# Run this once to set up the training environment on M3 Ultra
#

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}"
cat << "EOF"
   ____                _ _ _
  / ___|___  _ __  ___(_) (_) ___ _ __   ___ ___
 | |   / _ \| '_ \/ __| | | |/ _ \ '_ \ / __/ _ \
 | |__| (_) | | | \__ \ | | |  __/ | | | (_|  __/
  \____\___/|_| |_|___/_|_|_|\___|_| |_|\___\___|

  Training Environment Setup

EOF
echo -e "${NC}"

PROJECT_DIR="/Users/thistlequell/git/consilience"
TRAINING_DIR="$PROJECT_DIR/training"

cd $PROJECT_DIR

echo -e "${GREEN}Step 1: Creating directory structure...${NC}"
mkdir -p training/data/{raw,processed,synthetic,splits}
mkdir -p training/checkpoints/{phase4_pretrain,theory_of_mind,relationships,best}
mkdir -p training/logs
mkdir -p training/scripts
echo "  ✓ Directories created"

echo ""
echo -e "${GREEN}Step 2: Making scripts executable...${NC}"
chmod +x training/train.sh
chmod +x training/status.sh
chmod +x training/scripts/*.py
echo "  ✓ Scripts are now executable"

echo ""
echo -e "${GREEN}Step 3: Checking Python dependencies...${NC}"

# Check for required packages
python3 -c "import mlx.core" 2>/dev/null && echo "  ✓ MLX installed" || echo "  ⚠ MLX not found (install: pip3 install mlx)"
python3 -c "import numpy" 2>/dev/null && echo "  ✓ NumPy installed" || echo "  ⚠ NumPy not found (install: pip3 install numpy)"
python3 -c "import tqdm" 2>/dev/null && echo "  ✓ tqdm installed" || echo "  ⚠ tqdm not found (install: pip3 install tqdm)"

echo ""
echo -e "${GREEN}Step 4: Checking for tmux...${NC}"
if command -v tmux &> /dev/null; then
    echo "  ✓ tmux installed"
else
    echo "  ⚠ tmux not found (install: brew install tmux)"
    echo "    (Optional but recommended for persistent sessions)"
fi

echo ""
echo -e "${GREEN}Step 5: Testing data generation...${NC}"
echo "  Generating 100 test examples..."
python3 training/scripts/00_generate_synthetic_data.py --test 2>&1 | tail -n 5

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}✓ Setup Complete!${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "Ready to train! Next steps:"
echo ""
echo "  1. Start training:"
echo -e "     ${YELLOW}./training/train.sh${NC}"
echo ""
echo "  2. Or start in tmux (recommended):"
echo -e "     ${YELLOW}tmux new -s training${NC}"
echo -e "     ${YELLOW}./training/train.sh${NC}"
echo ""
echo "  3. Check status anytime:"
echo -e "     ${YELLOW}./training/status.sh${NC}"
echo ""
echo "Training will take ~1-2 days. You can disconnect and it'll keep running."
echo ""
