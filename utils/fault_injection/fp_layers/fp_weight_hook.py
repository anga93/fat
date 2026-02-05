"""
Floating-point weight fault injection hooks for standard PyTorch layers.
"""

import torch
import torch.nn as nn
from typing import Optional, Literal
from ..fp_bit_flip_utils import inject_bit_faults_tensor, Region
from ..statistics import LayerStatistics


class FPWeightFaultInjectionHook:
    """
    Forward pre-hook for injecting bit-level faults into FP32/FP16 weights.

    This hook is registered on Conv2d, Linear, etc. layers to corrupt weights
    before the forward pass using IEEE-754 bit flips.
    """

    def __init__(
        self,
        module: nn.Module,
        probability: float = 0.01,
        precision: str = "fp32",
        region: Region = "mantissa",
        bit_position: Optional[int] = None,
        enabled: bool = True,
        apply_during_train: bool = True,
        apply_during_eval: bool = False,
        layer_id: Optional[int] = None,
        track_statistics: bool = False,
    ):
        """
        Parameters
        ----------
        module : nn.Module
            The layer whose weights will be corrupted (Conv2d, Linear, etc.).
        probability : float
            Fault injection probability [0.0, 1.0].
        precision : {"fp32", "fp16"}
            Floating-point precision.
        region : {"sign", "exponent", "mantissa"}
            IEEE-754 region to target.
        bit_position : int or None
            Specific bit within region (None = random).
        enabled : bool
            Master switch for injection.
        apply_during_train : bool
            Inject faults during training.
        apply_during_eval : bool
            Inject faults during evaluation.
        layer_id : int or None
            Unique identifier for this layer (for statistics).
        track_statistics : bool
            Whether to track fault statistics.
        """
        self.module = module
        self.probability = probability
        self.precision = precision
        self.region = region
        self.bit_position = bit_position
        self.enabled = enabled
        self.apply_during_train = apply_during_train
        self.apply_during_eval = apply_during_eval
        self.layer_id = layer_id
        self.track_statistics = track_statistics

        self.statistics: Optional[LayerStatistics] = None
        if track_statistics:
            self.statistics = LayerStatistics()

        # Cache for original weights (optional, for restoration)
        self.original_weight: Optional[torch.Tensor] = None
        self.original_bias: Optional[torch.Tensor] = None

    def __call__(self, module: nn.Module, input: tuple) -> None:
        """
        Hook function called before forward pass.

        Modifies module.weight (and optionally module.bias) in-place.
        """
        # Check if injection should be applied
        should_inject = self.enabled and (
            (module.training and self.apply_during_train) or
            (not module.training and self.apply_during_eval)
        )

        if not should_inject or self.probability <= 0.0:
            return

        # Inject faults into weights
        if hasattr(module, 'weight') and module.weight is not None:
            self._inject_weight_faults(module)

        # Optionally inject into bias (usually not needed for hardware faults)
        # if hasattr(module, 'bias') and module.bias is not None:
        #     self._inject_bias_faults(module)

    def _inject_weight_faults(self, module: nn.Module):
        """Inject bit-level faults into weight tensor."""
        weight = module.weight

        # Generate fault mask
        mask = torch.rand_like(weight) < self.probability

        if not mask.any():
            # No faults to inject
            return

        # Store clean weights for statistics
        if self.track_statistics and self.statistics is not None:
            clean_weight = weight.clone()

        # Apply bit-level faults
        with torch.no_grad():
            faulty_weight = inject_bit_faults_tensor(
                x=weight,
                mask=mask,
                precision=self.precision,
                region=self.region,
                bit_position=self.bit_position,
                inplace=True,  # Modify in-place
            )

        # Track statistics if enabled
        if self.track_statistics and self.statistics is not None:
            self._record_statistics(clean_weight, faulty_weight, mask)

    def _record_statistics(
        self,
        clean: torch.Tensor,
        faulty: torch.Tensor,
        mask: torch.Tensor,
    ):
        """Record fault injection statistics."""
        if self.statistics is None:
            return

        # Compute metrics
        total_elements = mask.numel()
        num_faults = mask.sum().item()

        # RMSE on faulty positions only
        if num_faults > 0:
            faulty_clean = clean[mask]
            faulty_values = faulty[mask]
            rmse = torch.sqrt(torch.mean((faulty_clean - faulty_values) ** 2)).item()

            # Cosine similarity (flatten tensors)
            clean_flat = clean.flatten()
            faulty_flat = faulty.flatten()
            cos_sim = torch.nn.functional.cosine_similarity(
                clean_flat.unsqueeze(0),
                faulty_flat.unsqueeze(0),
            ).item()
        else:
            rmse = 0.0
            cos_sim = 1.0

        # Update statistics
        self.statistics.total_activations += total_elements  # Using same counter for consistency
        self.statistics.total_faults += num_faults
        self.statistics.rmse_sum += rmse
        self.statistics.cosine_similarity_sum += cos_sim
        self.statistics.sample_count += 1

    def cache_original_weights(self):
        """Store original weights before any injection (for restoration)."""
        if hasattr(self.module, 'weight') and self.module.weight is not None:
            self.original_weight = self.module.weight.clone()
        if hasattr(self.module, 'bias') and self.module.bias is not None:
            self.original_bias = self.module.bias.clone()

    def restore_original_weights(self):
        """Restore cached original weights."""
        if self.original_weight is not None and hasattr(self.module, 'weight'):
            self.module.weight.data.copy_(self.original_weight)
        if self.original_bias is not None and hasattr(self.module, 'bias'):
            self.module.bias.data.copy_(self.original_bias)
