# Floating-Point Fault Injection Framework

This document describes the **FP32/FP16 fault injection framework** integrated into the FAT (Fault-Aware Training) project. This framework enables IEEE-754 bit-level fault injection for full-precision PyTorch models.

---

## Overview

The FP fault injection framework extends the existing quantized fault injection system to support **non-quantized floating-point models** (FP32, FP16). It implements bit-level fault injection using IEEE-754 manipulation, adapted from the MXformats-vs-floatingpoint project.

### Key Features

✅ **IEEE-754 Bit-Level Manipulation**: Direct bit flipping in floating-point representation
✅ **Region-Specific Faults**: Target sign, exponent, or mantissa bits independently
✅ **Dual Precision Support**: FP32 and FP16
✅ **Activation & Weight Injection**: Inject faults in both activations and weights
✅ **YAML Configurable**: Easy experiment setup via config files
✅ **Statistics Tracking**: Monitor fault impact with RMSE and cosine similarity
✅ **Unified Framework**: Works alongside quantized fault injection

---

## Architecture

### Components

```
utils/fault_injection/
├── fp_bit_flip_utils.py              # Core IEEE-754 manipulation (from MXformats)
├── fp_activation_injector.py         # Activation injector for FP models
├── fp_weight_injector.py             # Weight injector for FP models
├── fp_layers/
│   ├── fp_activation_layer.py        # FP activation fault injection layer
│   └── fp_weight_hook.py             # FP weight fault injection hook
└── config.py                         # Extended with FP-specific parameters
```

### IEEE-754 Regions

**FP32 (32 bits)**:
- **Sign** (1 bit): Bit 31
- **Exponent** (8 bits): Bits 23-30
- **Mantissa** (23 bits): Bits 0-22

**FP16 (16 bits)**:
- **Sign** (1 bit): Bit 15
- **Exponent** (5 bits): Bits 10-14
- **Mantissa** (10 bits): Bits 0-9

---

## Usage

### 1. Quick Start

```python
import torch
import torch.nn as nn
from utils.fault_injection import (
    FaultInjectionConfig,
    FPActivationInjector,
    FaultStatistics,
)

# Create your FP32 model
model = nn.Sequential(
    nn.Conv2d(3, 64, 3),
    nn.ReLU(),
    nn.Linear(64, 10),
)

# Configure FP fault injection
config = FaultInjectionConfig(
    enabled=True,
    target_type="activation",
    probability=5.0,              # 5% of activations
    apply_during="train",
    fp_precision="fp32",
    fp_target_region="exponent",  # High-impact faults
    fp_bit_position=None,         # Random bit
    track_statistics=True,
)

# Inject fault layers
injector = FPActivationInjector()
model = injector.inject(model, config)

# Setup statistics (optional)
stats = FaultStatistics(num_layers=injector.get_num_layers(model))
injector.set_statistics(model, stats)

# Train normally
for epoch in range(epochs):
    # ... training loop ...
    pass

# View statistics
stats.print_report()
```

### 2. YAML Configuration

**Example: FP32 Activation Fault Injection**

```yaml
# configs/fp32_activation_fault_injection.yaml

fp_activation_fault_injection:
  enabled: true
  target_type: activation
  probability: 5.0
  apply_during: "train"           # "train", "eval", or "both"

  # Target standard PyTorch layers
  target_layers:
    - "Conv2d"
    - "Linear"
    - "ReLU"
    - "BatchNorm2d"

  # FP-specific parameters
  fp_precision: "fp32"            # "fp32" or "fp16"
  fp_target_region: "exponent"    # "sign", "exponent", "mantissa", "random"
  fp_bit_position: null           # null = random, or specific bit index

  track_statistics: true
  verbose: true
```

**Run training:**

```bash
python train.py --config configs/fp32_activation_fault_injection.yaml
```

### 3. Weight Fault Injection

```yaml
fp_weight_fault_injection:
  enabled: true
  target_type: weight
  probability: 2.0
  apply_during: "both"

  target_layers:
    - "Conv2d"
    - "Linear"

  fp_precision: "fp32"
  fp_target_region: "mantissa"    # Lower impact than exponent
  fp_bit_position: null

  track_statistics: true
  verbose: false
```

### 4. FP16 Mixed-Precision

```yaml
training:
  use_amp: true                   # Enable automatic mixed precision

fp_activation_fault_injection:
  enabled: true
  target_type: activation
  probability: 10.0
  apply_during: "eval"

  fp_precision: "fp16"            # Match model precision
  fp_target_region: "mantissa"

  track_statistics: true
```

---

## Configuration Reference

### FP-Specific Parameters

| Parameter | Type | Values | Description |
|-----------|------|--------|-------------|
| `fp_precision` | string | `"fp32"`, `"fp16"` | Floating-point format |
| `fp_target_region` | string | `"sign"`, `"exponent"`, `"mantissa"`, `"random"` | IEEE-754 region to target |
| `fp_bit_position` | int or null | `0-31` (fp32), `0-15` (fp16), `null` | Specific bit or random |

### Standard Parameters

| Parameter | Type | Values | Description |
|-----------|------|--------|-------------|
| `enabled` | bool | `true`, `false` | Master switch |
| `target_type` | string | `"activation"`, `"weight"` | What to inject into |
| `probability` | float | `0.0-100.0` | Fault rate (%) |
| `apply_during` | string | `"train"`, `"eval"`, `"both"` | When to inject |
| `target_layers` | list | Layer names | Which layers to target |
| `track_statistics` | bool | `true`, `false` | Enable metrics |
| `verbose` | bool | `true`, `false` | Print debug info |

---

## Fault Impact by Region

### Sign Bit
- **Impact**: Catastrophic (flips sign of entire number)
- **Use case**: Simulating worst-case faults
- **Example**: `3.14 → -3.14`

### Exponent Bits
- **Impact**: High (changes magnitude dramatically)
- **Use case**: Simulating severe hardware faults
- **Example**: `1.0 → 2.0` or `1.0 → 0.5`

### Mantissa Bits
- **Impact**: Low to medium (precision loss)
- **Use case**: Realistic memory bit flips
- **Example**: `3.14159 → 3.14161` (small change)

---

## Example Experiments

### Experiment 1: Region Comparison

```yaml
# Run 3 experiments with different regions
fp_target_region: "sign"       # Exp 1
fp_target_region: "exponent"   # Exp 2
fp_target_region: "mantissa"   # Exp 3
```

**Expected results**:
- Sign faults → Lowest accuracy
- Exponent faults → Medium accuracy
- Mantissa faults → Highest accuracy

### Experiment 2: Fault-Aware Training

```yaml
fp_activation_fault_injection:
  enabled: true
  apply_during: "train"        # Inject during training
  probability: 5.0
  fp_target_region: "exponent"
```

Train model with faults, then evaluate on clean data. Compare to:
1. Baseline (no faults)
2. Inject-only-at-eval (faults only during evaluation)

### Experiment 3: Probability Sweep

```python
# Sweep fault probabilities: 0%, 1%, 5%, 10%, 20%, 50%
for prob in [0, 1, 5, 10, 20, 50]:
    config.probability = prob
    # Train and evaluate
```

Plot: Accuracy vs Fault Probability

---

## Testing

Run the test suite to verify the integration:

```bash
python test_fp_fault_injection.py
```

**Tests included**:
1. FP32 activation injection
2. FP16 weight injection
3. Region impact comparison
4. Dynamic probability adjustment

---

## Comparison: Quantized vs Floating-Point

| Feature | Quantized Injection | FP Injection |
|---------|---------------------|--------------|
| Target models | Brevitas quantized | Standard PyTorch |
| Precision | INT8, INT4, etc. | FP32, FP16 |
| Bit manipulation | Integer operations | IEEE-754 bit flip |
| Strategies | Random, LSB/MSB flip | Region-specific |
| Target layers | `QuantConv2d`, `QuantLinear` | `Conv2d`, `Linear`, `ReLU` |
| Use case | Quantized deployment | Full-precision deployment |

**Both can be used simultaneously** in hybrid quantized/FP models!

---

## Advanced Usage

### Dynamic Probability Adjustment

```python
# Adjust probability during training
for epoch in range(epochs):
    if epoch < 10:
        injector.update_probability(model, 2.0)   # Low during warmup
    else:
        injector.update_probability(model, 10.0)  # Higher later

    # Train epoch
    train_one_epoch(...)
```

### Enable/Disable at Runtime

```python
# Disable injection temporarily
injector.set_enabled(model, False)
# Evaluate clean accuracy
clean_acc = evaluate(model, test_loader)

# Re-enable
injector.set_enabled(model, True)
# Evaluate faulty accuracy
faulty_acc = evaluate(model, test_loader)
```

### Per-Layer Probability

```python
# Update only specific layer
injector.update_probability(model, probability=20.0, layer_id=3)
```

---

## Integration with MXformats Project

This framework ports and extends the bit-flip logic from your MXformats-vs-floatingpoint project:

**From MXformats** (`fault_injection.py`):
- `flip_bit_in_float32()` → `fp_bit_flip_utils.flip_bit_in_float32()`
- `flip_bit_in_float16()` → `fp_bit_flip_utils.flip_bit_in_float16()`
- `inject_random_bit_faults_tensor()` → `fp_bit_flip_utils.inject_bit_faults_tensor()`
- Region definitions (REGIONS_FP32, REGIONS_FP16)

**Extensions in FAT**:
- PyTorch `nn.Module` integration
- Injector pattern (activation/weight)
- YAML configuration
- Statistics tracking
- Trainer integration
- Compatible with existing FAT framework

---

## Troubleshooting

### Issue: "No layers were injected"

**Cause**: Model uses quantized layers, but FP injector targets standard layers.

**Solution**: Use `ActivationFaultInjector` for quantized models, `FPActivationInjector` for FP models.

### Issue: "dtype mismatch" with FP16

**Cause**: Model is FP16 but config specifies `fp_precision: "fp32"`.

**Solution**: Match `fp_precision` to model dtype:
```yaml
fp_precision: "fp16"  # For FP16 models
```

### Issue: Statistics show 0 faults

**Cause**: `apply_during` doesn't match training phase.

**Solution**: Set `apply_during: "both"` or match to your eval phase.

---

## Citation

If you use this framework in your research, please cite:

```bibtex
@misc{fat_fp_injection_2026,
  title={Floating-Point Fault Injection Framework for FAT},
  author={[Your Name]},
  year={2026},
  note={Integrated from MXformats-vs-floatingpoint project}
}
```

---

## Future Work

Potential extensions:
- [ ] BFloat16 support
- [ ] Multi-bit faults (flip multiple bits simultaneously)
- [ ] Stuck-at faults (permanent bit stuck at 0 or 1)
- [ ] Time-dependent fault rates (increasing over mission duration)
- [ ] Spatial correlation (nearby weights more likely to fault)

---

## Support

For questions or issues:
1. Check example configs in `configs/`
2. Run `test_fp_fault_injection.py`
3. Review this README
4. Open an issue on GitHub

---

**Happy Fault-Aware Training! 🚀**
