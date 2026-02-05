#!/usr/bin/env python3
"""Quick check to verify MPS (Metal Performance Shaders) is available."""

import torch

print(f"PyTorch version: {torch.__version__}")
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")

if torch.backends.mps.is_available():
    device = torch.device("mps")
    x = torch.randn(1000, 1000, device=device)
    y = torch.randn(1000, 1000, device=device)
    z = x @ y
    print(f"✅ MPS is working! Successfully ran matrix multiplication on GPU")
else:
    print("❌ MPS not available. Will fall back to CPU.")
