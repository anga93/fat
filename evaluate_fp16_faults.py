#!/usr/bin/env python3
"""
Evaluate FP16 model accuracy with and without fault injection.

Tests mantissa bit-flip faults on FP16 inference.

Usage:
    python evaluate_fp16_faults.py --model resnet18 --dataset cifar10 --fault-prob 5.0
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import get_dataset
from models import get_model
from utils.fault_injection import (
    FaultInjectionConfig,
    FPActivationInjector,
    FaultStatistics,
)


def evaluate(model: nn.Module, loader: DataLoader, device: str = "cuda") -> float:
    """
    Evaluate model accuracy on dataset.

    Args:
        model: Model to evaluate
        loader: DataLoader for evaluation
        device: Device to run on

    Returns:
        Accuracy as percentage (0-100)
    """
    model.eval()
    correct = 0
    total = 0

    # Detect model dtype (FP16 or FP32)
    model_dtype = next(model.parameters()).dtype

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating"):
            images = images.to(device, dtype=model_dtype)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = outputs.max(1)

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100.0 * correct / total
    return accuracy


def main():
    parser = argparse.ArgumentParser(description="Evaluate FP16 with fault injection")
    parser.add_argument("--model", type=str, default="resnet18", help="Model architecture")
    parser.add_argument("--dataset", type=str, default="cifar10", help="Dataset name")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--fault-prob", type=float, default=5.0, help="Fault probability (%)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path")
    parser.add_argument("--device", type=str, default="auto", help="Device (cuda/mps/cpu/auto)")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")

    args = parser.parse_args()

    print("=" * 80)
    print("FP16 Fault Injection Evaluation")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Fault probability: {args.fault_prob}%")
    print(f"Device: {args.device}")
    print()

    # Set device (support MPS for Mac M1/M2/M3/M4)
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif args.device == "auto":
        # Auto-detect best device
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # Create unified config
    num_classes = 10 if args.dataset in ["cifar10", "mnist", "fashion_mnist"] else 100
    config = {
        "dataset": {
            "name": args.dataset,
            "root": "./data",
            "num_workers": args.num_workers,
            "download": True,
        },
        "training": {
            "batch_size": args.batch_size,
        },
        "model": {
            "name": args.model,
            "num_classes": num_classes,
            "in_channels": 3,
        }
    }

    # Load dataset
    print("Loading dataset...")
    dataset = get_dataset(config)
    _, _, test_loader = dataset.get_loaders()  # train, val, test

    # Load model
    print(f"Loading {args.model} model...")
    model = get_model(config)

    # Load checkpoint if provided
    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

    # Move to device first
    print(f"Moving model to {device}...")
    model = model.to(device)

    # Convert to FP16 (supported on CUDA and MPS)
    if device.type in ["cuda", "mps"]:
        print("Converting model to FP16...")
        model = model.half()
        print(f"✓ Model converted to FP16 on {device}")
    else:
        print("⚠ Warning: FP16 not fully supported on CPU, using FP32")
        print("  For FP16 testing, use --device mps (Mac M1/M2/M3/M4) or --device cuda")

    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
    print()

    # ========================================
    # BASELINE EVALUATION (No faults)
    # ========================================
    print("=" * 80)
    print("BASELINE EVALUATION (No Faults)")
    print("=" * 80)

    baseline_acc = evaluate(model, test_loader, device)
    print(f"\n✓ Baseline Accuracy: {baseline_acc:.2f}%\n")

    # ========================================
    # FAULTY EVALUATION (Mantissa faults)
    # ========================================
    print("=" * 80)
    print("FAULTY EVALUATION (Mantissa Bit-Flips)")
    print("=" * 80)

    # Configure FP16 fault injection on mantissa
    config = FaultInjectionConfig(
        enabled=True,
        target_type="activation",
        probability=args.fault_prob,
        apply_during="eval",  # Only during evaluation
        target_layers=["Conv2d", "Linear", "ReLU"],  # Standard PyTorch layers
        fp_precision="fp16",  # FP16 format
        fp_target_region="mantissa",  # Target mantissa bits
        fp_bit_position=None,  # Random bit within mantissa
        track_statistics=False,  # Disabled for speed
        verbose=False,
    )

    # Inject fault layers
    print(f"Injecting FP16 fault layers (mantissa, {args.fault_prob}% probability)...")
    injector = FPActivationInjector()
    model = injector.inject(model, config)

    # Count injected layers
    num_layers = injector.get_num_layers(model)
    print(f"Injected {num_layers} fault injection layers\n")

    # Evaluate with faults
    faulty_acc = evaluate(model, test_loader, device)
    print(f"\n✓ Faulty Accuracy: {faulty_acc:.2f}%\n")

    # ========================================
    # SUMMARY
    # ========================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Model: {args.model} (FP16)")
    print(f"Dataset: {args.dataset}")
    print(f"Fault Configuration:")
    print(f"  - Precision: FP16")
    print(f"  - Target Region: Mantissa")
    print(f"  - Probability: {args.fault_prob}%")
    print()
    print(f"Results:")
    print(f"  Baseline Accuracy:  {baseline_acc:.2f}%")
    print(f"  Faulty Accuracy:    {faulty_acc:.2f}%")
    print(f"  Accuracy Drop:      {baseline_acc - faulty_acc:.2f}%")
    print(f"  Relative Drop:      {100 * (baseline_acc - faulty_acc) / baseline_acc:.2f}%")
    print("=" * 80)


if __name__ == "__main__":
    main()
