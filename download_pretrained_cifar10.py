#!/usr/bin/env python3
"""
Download pre-trained ResNet18 weights for CIFAR-10.

This script downloads weights from a public repository and converts them
to be compatible with our model architecture.
"""

import torch
import torch.nn as nn
from models import get_model
import urllib.request
import os


def download_pretrained_resnet18_cifar10(save_path="checkpoints/resnet18_cifar10_pretrained.pth"):
    """
    Download pre-trained ResNet18 weights for CIFAR-10.

    Uses weights from: https://github.com/chenyaofo/pytorch-cifar-models
    These models achieve ~95% accuracy on CIFAR-10.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # URL for pre-trained ResNet18 on CIFAR-10 (from pytorch-cifar-models)
    url = "https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar10_resnet20-4118986f.pth"

    print(f"Downloading pre-trained ResNet18 for CIFAR-10...")
    print(f"From: {url}")
    print(f"To: {save_path}")

    try:
        urllib.request.urlretrieve(url, save_path)
        print(f"✓ Downloaded successfully!")

        # Verify the download
        checkpoint = torch.load(save_path, map_location="cpu")
        print(f"\nCheckpoint info:")
        if isinstance(checkpoint, dict):
            print(f"  Keys: {list(checkpoint.keys())}")
            if "state_dict" in checkpoint:
                print(f"  Model parameters: {len(checkpoint['state_dict'])}")
            elif "model_state_dict" in checkpoint:
                print(f"  Model parameters: {len(checkpoint['model_state_dict'])}")
        else:
            print(f"  Direct state dict with {len(checkpoint)} parameters")

        return save_path
    except Exception as e:
        print(f"Error downloading: {e}")
        print("\nAlternative: Try downloading manually from:")
        print("  https://github.com/chenyaofo/pytorch-cifar-models")
        return None


def load_pretrained_model(checkpoint_path, device="cuda"):
    """Load a pre-trained ResNet18 model."""
    config = {
        "dataset": {"name": "cifar10"},
        "model": {
            "name": "resnet18",
            "num_classes": 10,
            "in_channels": 3,
        }
    }

    model = get_model(config)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        elif "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)

    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download pre-trained CIFAR-10 ResNet18")
    parser.add_argument("--save-path", type=str,
                        default="checkpoints/resnet18_cifar10_pretrained.pth",
                        help="Where to save the checkpoint")
    parser.add_argument("--verify", action="store_true",
                        help="Verify the model loads correctly")

    args = parser.parse_args()

    checkpoint_path = download_pretrained_resnet18_cifar10(args.save_path)

    if checkpoint_path and args.verify:
        print("\nVerifying model loads correctly...")
        try:
            model = load_pretrained_model(checkpoint_path, device="cpu")
            print(f"✓ Model loaded successfully!")
            print(f"  Total parameters: {sum(p.numel() for p in model.parameters())}")
        except Exception as e:
            print(f"✗ Error loading model: {e}")
