"""Fault injection framework for quantized and floating-point neural networks.

This module provides tools for injecting activation and weight faults into quantized
and full-precision neural networks to enable fault-aware training (FAT) and fault
resilience evaluation.

Main Components:
    - FaultInjectionConfig: Configuration dataclass for fault injection parameters (supports activation and weight)
    - ActivationFaultInjector: Runtime model transformer that adds activation fault injection layers (quantized)
    - WeightFaultInjector: Runtime model transformer that adds weight fault injection hooks (quantized)
    - FPActivationInjector: Activation fault injector for FP32/FP16 models (full-precision)
    - FPWeightInjector: Weight fault injector for FP32/FP16 models (full-precision)
    - FaultInjector: Backward compatibility alias for ActivationFaultInjector (deprecated)
    - FaultStatistics: Statistics tracking for injection analysis
    - QuantActivationFaultInjectionLayer: Layer that injects activation faults into QuantTensor activations

Injection Strategies:
    - RandomStrategy: Adds random values to activations/weights
    - LSBFlipStrategy: Flips least significant bits
    - MSBFlipStrategy: Flips most significant bits
    - FullFlipStrategy: Flips all bits

Floating-Point Bit-Level Injection:
    - FPActivationFaultInjectionLayer: IEEE-754 bit-flip for FP32/FP16 activations
    - FPWeightFaultInjectionHook: IEEE-754 bit-flip for FP32/FP16 weights
    - Supports region-specific faults (sign, exponent, mantissa)

Example:
    ```python
    from utils.fault_injection import (
        FaultInjectionConfig,
        ActivationFaultInjector,
        WeightFaultInjector,
        FaultStatistics,
    )

    # Create activation configuration
    act_config = FaultInjectionConfig(
        enabled=True,
        target_type="activation",
        probability=5.0,
        injection_type="random",
        apply_during="train",
    )

    # Inject activation fault layers into model
    act_injector = ActivationFaultInjector()
    model = act_injector.inject(model, act_config)

    # Create weight configuration
    weight_config = FaultInjectionConfig(
        enabled=True,
        target_type="weight",
        probability=2.0,
        injection_type="lsb_flip",
        apply_during="both",
    )

    # Inject weight fault hooks into model
    weight_injector = WeightFaultInjector()
    model = weight_injector.inject(model, weight_config)

    # Optional: Track statistics
    stats = FaultStatistics()
    act_injector.set_statistics(model, stats)
    weight_injector.set_statistics(model, stats)

    # Training loop
    for epoch in range(epochs):
        # ... train ...

    # Get statistics report
    stats.print_report()
    ```
"""

from __future__ import annotations

from .base_injector import BaseFaultInjector
from .config import FaultInjectionConfig
from .activation_injector import ActivationFaultInjector
from .weight_injector import WeightFaultInjector
from .activations.activation_functions import ActivationFaultInjectionFunction
from .activations.activation_layers import QuantActivationFaultInjectionLayer
from .statistics import FaultStatistics, LayerStatistics
from .strategies import (
    InjectionStrategy,
    RandomStrategy,
    LSBFlipStrategy,
    MSBFlipStrategy,
    FullFlipStrategy,
    get_strategy,
)
from .weights.weight_hooks import WeightFaultInjectionHook

# Floating-point injectors and utilities
from .fp_activation_injector import FPActivationInjector
from .fp_weight_injector import FPWeightInjector
from .fp_layers.fp_activation_layer import FPActivationFaultInjectionLayer
from .fp_layers.fp_weight_hook import FPWeightFaultInjectionHook
from . import fp_bit_flip_utils

# Backward compatibility alias (deprecated)
FaultInjector = ActivationFaultInjector

__all__ = [
    # Configuration
    "FaultInjectionConfig",
    # Base classes
    "BaseFaultInjector",
    # Quantized model injectors
    "ActivationFaultInjector",
    "WeightFaultInjector",
    "FaultInjector",  # Backward compatibility (deprecated)
    # Floating-point model injectors
    "FPActivationInjector",
    "FPWeightInjector",
    # Layers
    "QuantActivationFaultInjectionLayer",
    "FPActivationFaultInjectionLayer",
    # Activation injection components
    "ActivationFaultInjectionFunction",
    # Statistics
    "FaultStatistics",
    "LayerStatistics",
    # Strategies
    "InjectionStrategy",
    "RandomStrategy",
    "LSBFlipStrategy",
    "MSBFlipStrategy",
    "FullFlipStrategy",
    "get_strategy",
    # Weight injection components
    "WeightFaultInjectionHook",
    "WeightFaultInjectionFunction",
    "FPWeightFaultInjectionHook",
    # Utilities
    "fp_bit_flip_utils"
]
