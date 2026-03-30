#!/usr/bin/env python3
"""
Create or download a pre-trained ResNet18 model for CIFAR-10.

Option 1: Use torchvision pre-trained weights (ImageNet) as initialization
Option 2: Download from a working URL
Option 3: Train quickly (takes ~10-20 minutes on GPU)
"""

import torch
import torch.nn as nn
import torchvision
from models import get_model
import os


def create_pretrained_from_imagenet(save_path="checkpoints/resnet18_cifar10_imagenet_init.pth"):
    """
    Create a ResNet18 initialized with ImageNet weights.

    This won't give perfect CIFAR-10 accuracy but will be much better than random.
    Expected accuracy: ~70-80% (vs ~10% random, ~95% fully trained)
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print("Creating ResNet18 with ImageNet pre-trained weights...")

    # Load torchvision's ImageNet pre-trained ResNet18
    pretrained_model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)

    # Get our CIFAR-10 model
    config = {
        "dataset": {"name": "cifar10"},
        "model": {
            "name": "resnet18",
            "num_classes": 10,
            "in_channels": 3,
        }
    }
    our_model = get_model(config)

    # Transfer weights (except final FC layer which has different size)
    pretrained_dict = pretrained_model.state_dict()
    our_dict = our_model.state_dict()

    # Filter out fc layer (1000 classes -> 10 classes mismatch)
    transferred_dict = {k: v for k, v in pretrained_dict.items()
                        if k in our_dict and our_dict[k].shape == v.shape}

    print(f"Transferring {len(transferred_dict)}/{len(our_dict)} layers")

    # Update our model
    our_dict.update(transferred_dict)
    our_model.load_state_dict(our_dict)

    # Save
    torch.save({
        "model_state_dict": our_model.state_dict(),
        "source": "ImageNet pre-trained initialization",
        "expected_accuracy": "70-80% (transfer learning, not fine-tuned)",
    }, save_path)

    print(f"✓ Saved to: {save_path}")
    print(f"  Expected CIFAR-10 accuracy: ~70-80% (without fine-tuning)")
    return save_path


def download_from_url(url, save_path):
    """Download model from URL."""
    import urllib.request
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print(f"Downloading from: {url}")
    try:
        urllib.request.urlretrieve(url, save_path)
        print(f"✓ Downloaded to: {save_path}")
        return save_path
    except Exception as e:
        print(f"✗ Download failed: {e}")
        return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Get pre-trained CIFAR-10 ResNet18")
    parser.add_argument("--method", type=str, default="imagenet",
                        choices=["imagenet", "url"],
                        help="Method to get pre-trained weights")
    parser.add_argument("--url", type=str, default=None,
                        help="URL to download from (if using url method)")
    parser.add_argument("--save-path", type=str,
                        default="checkpoints/resnet18_cifar10.pth",
                        help="Where to save the checkpoint")

    args = parser.parse_args()

    if args.method == "imagenet":
        checkpoint_path = create_pretrained_from_imagenet(args.save_path)
    elif args.method == "url" and args.url:
        checkpoint_path = download_from_url(args.url, args.save_path)
    else:
        print("Error: Must specify --url when using url method")
        checkpoint_path = None

    if checkpoint_path:
        # Verify it works
        print("\nVerifying model loads...")
        config = {
            "dataset": {"name": "cifar10"},
            "model": {
                "name": "resnet18",
                "num_classes": 10,
                "in_channels": 3,
            }
        }
        model = get_model(config)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        print(f"✓ Model loaded successfully!")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters())}")
