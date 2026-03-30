#!/bin/bash
# Sweep fault probabilities for FP16 mantissa faults

MODEL="resnet18"
DATASET="cifar10"
CHECKPOINT=""  # Optional: path to checkpoint
DEVICE="auto"  # auto, cuda, mps, or cpu

echo "FP16 Mantissa Fault Probability Sweep"
echo "======================================"
echo ""

# Test different fault probabilities
for PROB in 0.0 1.0 5.0 10.0 20.0 50.0; do
    echo "Testing probability: ${PROB}%"
    python evaluate_fp16_faults.py \
        --model $MODEL \
        --dataset $DATASET \
        --fault-prob $PROB \
        --device $DEVICE \
        ${CHECKPOINT:+--checkpoint $CHECKPOINT} \
        | tee results_fp16_mantissa_p${PROB}.log
    echo ""
    echo "---"
    echo ""
done

echo "Done! Check results_fp16_mantissa_p*.log files"
