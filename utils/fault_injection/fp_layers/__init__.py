"""
Floating-point fault injection layers for standard PyTorch models.
"""

from .fp_activation_layer import FPActivationFaultInjectionLayer

__all__ = ["FPActivationFaultInjectionLayer"]
