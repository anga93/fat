"""
Floating-point activation fault injector for standard PyTorch models.
"""

import torch.nn as nn
from typing import List, Optional
from .base_injector import BaseFaultInjector
from .config import FaultInjectionConfig
from .fp_layers.fp_activation_layer import FPActivationFaultInjectionLayer
from .statistics import FaultStatistics


class _FPActivationFaultInjectionWrapper(nn.Module):
    """
    Wrapper that combines a layer with its FP fault injection layer.

    Forward flow: layer(x) -> FP_injection_layer(output) -> final_output
    """

    def __init__(self, layer: nn.Module, injection_layer: FPActivationFaultInjectionLayer):
        super().__init__()
        self.layer = layer
        self.injection_layer = injection_layer

    def forward(self, x):
        out = self.layer(x)
        return self.injection_layer(out)


class FPActivationInjector(BaseFaultInjector):
    """
    Activation fault injector for floating-point (FP32/FP16) models.

    Wraps standard PyTorch layers (Conv2d, Linear, ReLU, etc.) with
    FP bit-level fault injection layers that corrupt activations.

    Works with non-quantized models (full-precision PyTorch).
    """

    # Default target layers for FP models
    DEFAULT_TARGET_LAYERS = [
        "Conv2d",
        "Linear",
        "ReLU",
        "ReLU6",
        "LeakyReLU",
        "GELU",
        "SiLU",
        "BatchNorm2d",
    ]

    def inject(self, model: nn.Module, config: FaultInjectionConfig) -> nn.Module:
        """
        Inject FP fault injection layers into the model.

        Parameters
        ----------
        model : nn.Module
            PyTorch model (full-precision).
        config : FaultInjectionConfig
            Fault injection configuration.

        Returns
        -------
        nn.Module
            Modified model with fault injection layers.
        """
        if config.target_type != "activation":
            raise ValueError(f"FPActivationInjector requires target_type='activation', got '{config.target_type}'")

        # Determine target layer types
        target_layers = config.target_layers if config.target_layers else self.DEFAULT_TARGET_LAYERS

        # Convert probability from percentage to [0, 1]
        probability = config.probability / 100.0

        # Track injection
        layer_id = 0

        def _inject_recursive(module: nn.Module, prefix: str = ""):
            nonlocal layer_id

            # Iterate over immediate children
            children_to_replace = {}

            for name, child in module.named_children():
                full_name = f"{prefix}.{name}" if prefix else name

                # Check if this layer should be wrapped
                layer_type = child.__class__.__name__
                if layer_type in target_layers:
                    # Create fault injection layer
                    injection_layer = FPActivationFaultInjectionLayer(
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

                    # Wrap layer
                    wrapper = _FPActivationFaultInjectionWrapper(child, injection_layer)
                    children_to_replace[name] = wrapper

                    if config.verbose:
                        print(f"[FPActivationInjector] Wrapped layer: {full_name} ({layer_type})")

                    layer_id += 1

                else:
                    # Recursively process child
                    _inject_recursive(child, full_name)

            # Replace wrapped layers
            for name, wrapper in children_to_replace.items():
                setattr(module, name, wrapper)

        _inject_recursive(model)

        if config.verbose:
            print(f"[FPActivationInjector] Injected {layer_id} fault injection layers")

        return model

    def remove(self, model: nn.Module) -> nn.Module:
        """
        Remove FP fault injection layers from the model.

        Unwraps layers to restore original model structure.
        """
        def _remove_recursive(module: nn.Module):
            children_to_replace = {}

            for name, child in module.named_children():
                if isinstance(child, _FPActivationFaultInjectionWrapper):
                    # Unwrap: restore original layer
                    children_to_replace[name] = child.layer
                else:
                    # Recursively process child
                    _remove_recursive(child)

            # Replace unwrapped layers
            for name, original_layer in children_to_replace.items():
                setattr(module, name, original_layer)

        _remove_recursive(model)
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
            Model with injected layers.
        probability : float
            New probability (percentage 0-100).
        layer_id : int or None
            Target specific layer (None = all layers).
        """
        probability_normalized = probability / 100.0

        def _update_recursive(module: nn.Module):
            for child in module.children():
                if isinstance(child, _FPActivationFaultInjectionWrapper):
                    inj_layer = child.injection_layer
                    if layer_id is None or inj_layer.layer_id == layer_id:
                        inj_layer.probability = probability_normalized
                else:
                    _update_recursive(child)

        _update_recursive(model)

    def set_enabled(self, model: nn.Module, enabled: bool) -> None:
        """
        Enable or disable fault injection globally.

        Parameters
        ----------
        model : nn.Module
            Model with injected layers.
        enabled : bool
            True to enable, False to disable.
        """
        def _set_enabled_recursive(module: nn.Module):
            for child in module.children():
                if isinstance(child, _FPActivationFaultInjectionWrapper):
                    child.injection_layer.enabled = enabled
                else:
                    _set_enabled_recursive(child)

        _set_enabled_recursive(model)

    def set_statistics(self, model: nn.Module, statistics: FaultStatistics) -> None:
        """
        Attach statistics tracker to all injection layers.

        Parameters
        ----------
        model : nn.Module
            Model with injected layers.
        statistics : FaultStatistics
            Statistics tracker instance.
        """
        layer_id = 0

        def _set_statistics_recursive(module: nn.Module):
            nonlocal layer_id
            for child in module.children():
                if isinstance(child, _FPActivationFaultInjectionWrapper):
                    inj_layer = child.injection_layer
                    inj_layer.statistics = statistics.layers[layer_id]
                    layer_id += 1
                else:
                    _set_statistics_recursive(child)

        _set_statistics_recursive(model)

    def get_num_layers(self, model: nn.Module) -> int:
        """
        Count the number of injected layers.

        Parameters
        ----------
        model : nn.Module
            Model with injected layers.

        Returns
        -------
        int
            Number of injection layers.
        """
        count = 0

        def _count_recursive(module: nn.Module):
            nonlocal count
            for child in module.children():
                if isinstance(child, _FPActivationFaultInjectionWrapper):
                    count += 1
                else:
                    _count_recursive(child)

        _count_recursive(model)
        return count
