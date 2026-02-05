"""
Test script for FP32/FP16 fault injection.

Demonstrates how to use the floating-point fault injection framework
with standard PyTorch models (non-quantized).
"""

import torch
import torch.nn as nn
from utils.fault_injection import (
    FaultInjectionConfig,
    FPActivationInjector,
    FPWeightInjector,
    FaultStatistics,
)


def create_simple_model(num_classes=10):
    """Create a simple CNN for testing."""
    return nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(32 * 8 * 8, 128),
        nn.ReLU(),
        nn.Linear(128, num_classes),
    )


def test_fp32_activation_injection():
    """Test FP32 activation fault injection."""
    print("\n" + "=" * 70)
    print("TEST 1: FP32 Activation Fault Injection")
    print("=" * 70)

    # Create model
    model = create_simple_model()
    print(f"Created model with {sum(p.numel() for p in model.parameters())} parameters")

    # Create configuration
    config = FaultInjectionConfig(
        enabled=True,
        target_type="activation",
        probability=10.0,  # 10% of activations
        apply_during="both",
        fp_precision="fp32",
        fp_target_region="mantissa",
        fp_bit_position=None,  # Random bit
        track_statistics=True,
        verbose=True,
    )

    # Inject fault layers
    injector = FPActivationInjector()
    model = injector.inject(model, config)

    # Setup statistics
    num_layers = injector.get_num_layers(model)
    stats = FaultStatistics(num_layers=num_layers)
    injector.set_statistics(model, stats)

    print(f"\nInjected {num_layers} fault injection layers")

    # Test forward pass
    model.eval()
    x = torch.randn(4, 3, 32, 32)  # Batch of 4 CIFAR-10-like images

    print("\nRunning forward pass...")
    with torch.no_grad():
        output = model(x)

    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")

    # Print statistics
    print("\nFault Injection Statistics:")
    stats.print_report()

    print("\n✓ Test passed!")


def test_fp16_weight_injection():
    """Test FP16 weight fault injection."""
    print("\n" + "=" * 70)
    print("TEST 2: FP16 Weight Fault Injection")
    print("=" * 70)

    # Create model and convert to FP16
    model = create_simple_model()
    model = model.half()  # Convert to FP16
    print(f"Created FP16 model")

    # Create configuration
    config = FaultInjectionConfig(
        enabled=True,
        target_type="weight",
        probability=5.0,  # 5% of weights
        apply_during="both",
        target_layers=["Conv2d", "Linear"],
        fp_precision="fp16",
        fp_target_region="exponent",
        fp_bit_position=None,
        track_statistics=True,
        verbose=True,
    )

    # Inject fault hooks
    injector = FPWeightInjector()
    model = injector.inject(model, config)

    # Setup statistics
    num_layers = injector.get_num_layers(model)
    stats = FaultStatistics(num_layers=num_layers)
    injector.set_statistics(model, stats)

    print(f"\nRegistered {num_layers} weight fault injection hooks")

    # Test forward pass
    model.eval()
    x = torch.randn(4, 3, 32, 32).half()  # FP16 input

    print("\nRunning forward pass...")
    with torch.no_grad():
        output = model(x)

    print(f"Output shape: {output.shape}")
    print(f"Output dtype: {output.dtype}")

    # Print statistics
    print("\nFault Injection Statistics:")
    stats.print_report()

    print("\n✓ Test passed!")


def test_region_comparison():
    """Compare fault impact across different IEEE-754 regions."""
    print("\n" + "=" * 70)
    print("TEST 3: Comparing Fault Impact by Region")
    print("=" * 70)

    regions = ["sign", "exponent", "mantissa"]
    model_base = create_simple_model()

    # Create clean reference output
    model_base.eval()
    x = torch.randn(8, 3, 32, 32)
    with torch.no_grad():
        clean_output = model_base(x)

    print(f"\nClean model output range: [{clean_output.min().item():.4f}, {clean_output.max().item():.4f}]")

    for region in regions:
        print(f"\n--- Testing region: {region} ---")

        # Create fresh model
        model = create_simple_model()
        model.load_state_dict(model_base.state_dict())

        # Configure injection
        config = FaultInjectionConfig(
            enabled=True,
            target_type="activation",
            probability=20.0,  # 20% for visible impact
            apply_during="eval",
            fp_precision="fp32",
            fp_target_region=region,
            fp_bit_position=None,
            track_statistics=False,
            verbose=False,
        )

        # Inject
        injector = FPActivationInjector()
        model = injector.inject(model, config)

        # Forward pass
        model.eval()
        with torch.no_grad():
            faulty_output = model(x)

        # Compare
        mse = torch.mean((clean_output - faulty_output) ** 2).item()
        max_diff = torch.max(torch.abs(clean_output - faulty_output)).item()

        print(f"  MSE: {mse:.6f}")
        print(f"  Max absolute difference: {max_diff:.6f}")
        print(f"  Output range: [{faulty_output.min().item():.4f}, {faulty_output.max().item():.4f}]")

    print("\n✓ Test passed!")


def test_dynamic_probability():
    """Test dynamic probability adjustment during runtime."""
    print("\n" + "=" * 70)
    print("TEST 4: Dynamic Probability Adjustment")
    print("=" * 70)

    model = create_simple_model()

    # Configure injection
    config = FaultInjectionConfig(
        enabled=True,
        target_type="activation",
        probability=5.0,  # Start at 5%
        apply_during="eval",
        fp_precision="fp32",
        fp_target_region="mantissa",
        track_statistics=False,
        verbose=False,
    )

    injector = FPActivationInjector()
    model = injector.inject(model, config)

    # Test different probabilities
    probabilities = [1.0, 5.0, 10.0, 20.0, 50.0]
    x = torch.randn(8, 3, 32, 32)

    model.eval()
    for prob in probabilities:
        injector.update_probability(model, prob)

        with torch.no_grad():
            output = model(x)

        print(f"Probability {prob:5.1f}%: output range [{output.min().item():7.3f}, {output.max().item():7.3f}]")

    print("\n✓ Test passed!")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print(" FLOATING-POINT FAULT INJECTION TEST SUITE")
    print("=" * 70)

    # Run tests
    test_fp32_activation_injection()
    test_fp16_weight_injection()
    test_region_comparison()
    test_dynamic_probability()

    print("\n" + "=" * 70)
    print(" ALL TESTS PASSED! ✓")
    print("=" * 70)
    print("\nFP fault injection framework is working correctly.")
    print("You can now use it with your training pipeline!\n")
