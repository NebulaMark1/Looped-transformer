#!/bin/bash
# Run all three Looped Transformer experiments sequentially.
# Usage: bash run_all.sh

set -e

RESULTS_DIR="./results"
EPOCHS=15

echo "============================================"
echo "  Looped Transformer Experiment Suite"
echo "============================================"
echo ""

# Install dependencies (uncomment if needed)
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# pip install transformers datasets tqdm

echo "[1/3] Training baseline (traditional looped)..."
python train.py --mode baseline --epochs $EPOCHS --output_dir $RESULTS_DIR

echo ""
echo "[2/3] Training LoRA-per-Loop (r=8)..."
python train.py --mode lora --lora_rank 8 --epochs $EPOCHS --output_dir $RESULTS_DIR

echo ""
echo "[3/3] Training Full-Loop (independent params per loop)..."
python train.py --mode full --epochs $EPOCHS --output_dir $RESULTS_DIR

echo ""
echo "============================================"
echo "  All experiments complete!"
echo "============================================"
echo ""
echo "Analyze results:"
echo "  python analyze.py --results_dir $RESULTS_DIR"
echo "  python analyze.py --results_dir $RESULTS_DIR --detail"
