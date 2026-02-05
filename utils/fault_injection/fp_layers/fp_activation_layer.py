"""
Floating-point activation fault injection layer for standard PyTorch tensors.
"""

import torch
import torch.nn as nn
from typing import Optional, Literal
from ..fp_bit_flip_utils import inject_bit_faults_tensor, Region
from ..statistics import LayerStatistics


class FPActivationFaultInjectionLayer(nn.Module):
    """
    PyTorch layer that injects bit-level faults into FP32/FP16 activations.

    This layer is inserted after target layers (Conv2d, ReLU, etc.) to corrupt
    their output activations using IEEE-754 bit flips.

    Supports:
    - FP32 and FP16 precision
    - Region-specific faults (sign, exponent, mantissa)
    - Configurable fault probability
    - Optional statistics tracking
    """

    def __init__(
        self,
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
        probability : float
            Fault injection probability [0.0, 1.0].
            Example: 0.01 = 1% of activations corrupted.
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
        super().__init__()

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply fault injection to input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input activation tensor (fp32 or fp16).

        Returns
        -------
        torch.Tensor
            Tensor with injected faults (if conditions met).
        """
        # Check if injection should be applied
        should_inject = self.enabled and (
            (self.training and self.apply_during_train) or
            (not self.training and self.apply_during_eval)
        )

        if not should_inject or self.probability <= 0.0:
            return x

        # Generate fault mask (random selection based on probability)
        mask = torch.rand_like(x) < self.probability

        if not mask.any():
            # No faults to inject
            return x

        # Store clean values for statistics
        if self.track_statistics and self.statistics is not None:
            clean_x = x.clone()

        # Apply bit-level faults
        x_faulty = inject_bit_faults_tensor(
            x=x,
            mask=mask,
            precision=self.precision,
            region=self.region,
            bit_position=self.bit_position,
            inplace=False,
        )

        # Track statistics if enabled
        if self.track_statistics and self.statistics is not None:
            self._record_statistics(clean_x, x_faulty, mask)

        return x_faulty

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
        self.statistics.total_activations += total_elements
        self.statistics.total_faults += num_faults
        self.statistics.rmse_sum += rmse
        self.statistics.cosine_similarity_sum += cos_sim
        self.statistics.sample_count += 1

    def extra_repr(self) -> str:
        """String representation for layer printing."""
        return (
            f"probability={self.probability}, "
            f"precision={self.precision}, "
            f"region={self.region}, "
            f"enabled={self.enabled}"
        )
