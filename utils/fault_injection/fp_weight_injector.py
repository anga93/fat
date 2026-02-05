"""
Floating-point weight fault injector for standard PyTorch models.
"""

import torch.nn as nn
from typing import List, Optional, Dict
from .base_injector import BaseFaultInjector
from .config import FaultInjectionConfig
from .fp_layers.fp_weight_hook import FPWeightFaultInjectionHook
from .statistics import FaultStatistics


class FPWeightInjector(BaseFaultInjector):
    """
    Weight fault injector for floating-point (FP32/FP16) models.

    Registers forward pre-hooks on standard PyTorch layers (Conv2d, Linear, etc.)
    to inject FP bit-level faults into weights before forward pass.

    Works with non-quantized models (full-precision PyTorch).
    """

    # Default target layers for FP models
    DEFAULT_TARGET_LAYERS = [
        "Conv2d",
        "Linear",
        "ConvTranspose2d",
    ]

    def __init__(self):
        super().__init__()
        # Track registered hooks {layer_id: (module, hook_handle, hook_instance)}
        self.hooks: Dict[int, tuple] = {}

    def inject(self, model: nn.Module, config: FaultInjectionConfig) -> nn.Module:
        """
        Inject FP fault injection hooks into the model.

        Parameters
        ----------
        model : nn.Module
            PyTorch model (full-precision).
        config : FaultInjectionConfig
            Fault injection configuration.

        Returns
        -------
        nn.Module
            Modified model with fault injection hooks.
        """
        if config.target_type != "weight":
            raise ValueError(f"FPWeightInjector requires target_type='weight', got '{config.target_type}'")

        # Determine target layer types
        target_layers = config.target_layers if config.target_layers else self.DEFAULT_TARGET_LAYERS

        # Convert probability from percentage to [0, 1]
        probability = config.probability / 100.0

        # Track injection
        layer_id = 0

        def _inject_recursive(module: nn.Module, prefix: str = ""):
            nonlocal layer_id

            for name, child in module.named_children():
                full_name = f"{prefix}.{name}" if prefix else name

                # Check if this layer should get a hook
                layer_type = child.__class__.__name__
                if layer_type in target_layers:
                    # Create hook instance
                    hook = FPWeightFaultInjectionHook(
                        module=child,
                        probability=probability,
                        precision=config.fp_precision,
                        region=config.fp_target_region,
                        bit_position=config.fp_bit_position,
                        enabled=config.enabled,
                        apply_during_train=(config.apply_during in ["train", "both"]),
                        apply_during_eval=(config.apply_during in ["eval", "both"]),
                        layer_id=layer_id,
                        track_statistics=config.track_statistics,
                    )

                    # Register hook
                    hook_handle = child.register_forward_pre_hook(hook)

                    # Store hook info
                    self.hooks[layer_id] = (child, hook_handle, hook)

                    if config.verbose:
                        print(f"[FPWeightInjector] Registered hook on: {full_name} ({layer_type})")

                    layer_id += 1

                else:
                    # Recursively process child
                    _inject_recursive(child, full_name)

        _inject_recursive(model)

        if config.verbose:
            print(f"[FPWeightInjector] Registered {layer_id} fault injection hooks")

        return model

    def remove(self, model: nn.Module) -> nn.Module:
        """
        Remove FP fault injection hooks from the model.

        Unregisters all hooks to restore original model behavior.
        """
        for layer_id, (module, hook_handle, hook_instance) in self.hooks.items():
            hook_handle.remove()

        self.hooks.clear()
        return model

    def update_probability(
        self,
        model: nn.Module,
        probability: float,
        layer_id: Optional[int] = None,
    ) -> None:
        """
        Update fault injection probability dynamically.

        Parameters
        ----------
        model : nn.Module
            Model with injected hooks.
        probability : float
            New probability (percentage 0-100).
        layer_id : int or None
            Target specific layer (None = all layers).
        """
        probability_normalized = probability / 100.0

        if layer_id is not None:
            # Update specific layer
            if layer_id in self.hooks:
                _, _, hook_instance = self.hooks[layer_id]
                hook_instance.probability = probability_normalized
        else:
            # Update all layers
            for _, _, hook_instance in self.hooks.values():
                hook_instance.probability = probability_normalized

    def set_enabled(self, model: nn.Module, enabled: bool) -> None:
        """
        Enable or disable fault injection globally.

        Parameters
        ----------
        model : nn.Module
            Model with injected hooks.
        enabled : bool
            True to enable, False to disable.
        """
        for _, _, hook_instance in self.hooks.values():
            hook_instance.enabled = enabled

    def set_statistics(self, model: nn.Module, statistics: FaultStatistics) -> None:
        """
        Attach statistics tracker to all injection hooks.

        Parameters
        ----------
        model : nn.Module
            Model with injected hooks.
        statistics : FaultStatistics
            Statistics tracker instance.
        """
        for lid, (_, _, hook_instance) in self.hooks.items():
            hook_instance.statistics = statistics.layers[lid]

    def get_num_layers(self, model: nn.Module) -> int:
        """
        Count the number of injected hooks.

        Parameters
        ----------
        model : nn.Module
            Model with injected hooks.

        Returns
        -------
        int
            Number of injection hooks.
        """
        return len(self.hooks)
