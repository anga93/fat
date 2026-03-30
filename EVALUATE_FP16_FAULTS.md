# FP16 Fault Injection Evaluation Guide

Quick guide to evaluate FP16 models with mantissa bit-flip faults.

---

## Quick Start

### 1. Simple Evaluation (Baseline vs Faulty)

```bash
# Evaluate ResNet18 on CIFAR-10 with 5% mantissa faults
python evaluate_fp16_faults.py \
    --model resnet18 \
    --dataset cifar10 \
    --fault-prob 5.0
```

**Output:**
```
================================================================================
BASELINE EVALUATION (No Faults)
================================================================================
✓ Baseline Accuracy: 95.23%

================================================================================
FAULTY EVALUATION (Mantissa Bit-Flips)
================================================================================
✓ Faulty Accuracy: 92.15%

================================================================================
SUMMARY
================================================================================
  Baseline Accuracy:  95.23%
  Faulty Accuracy:    92.15%
  Accuracy Drop:      3.08%
  Relative Drop:      3.23%
```

---

## Usage Options

### Test with Pre-trained Checkpoint

```bash
python evaluate_fp16_faults.py \
    --model resnet18 \
    --dataset cifar10 \
    --fault-prob 10.0 \
    --checkpoint path/to/checkpoint.pth
```

### Different Datasets

```bash
# CIFAR-100
python evaluate_fp16_faults.py --model resnet18 --dataset cifar100

# MNIST
python evaluate_fp16_faults.py --model resnet18 --dataset mnist
```

### Different Models

```bash
# VGG16
python evaluate_fp16_faults.py --model vgg16 --dataset cifar10

# MobileNetV2
python evaluate_fp16_faults.py --model mobilenet_v2 --dataset cifar10
```

---

## Probability Sweep

Test multiple fault probabilities automatically:

```bash
# Run sweep: 0%, 1%, 5%, 10%, 20%, 50%
./scripts/test_fp16_fault_sweep.sh
```

Results saved to `results_fp16_mantissa_p*.log`

---

## Testing Different IEEE-754 Regions

### Mantissa Faults (Default)
```bash
python evaluate_fp16_faults.py --fault-prob 5.0
# Modify config.fp_target_region = "mantissa" in script
```

### Exponent Faults (High Impact)
Edit `evaluate_fp16_faults.py` line with `fp_target_region`:
```python
fp_target_region="exponent",  # Change from "mantissa"
```

### Sign Bit Faults (Catastrophic)
```python
fp_target_region="sign",
```

---

## Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model` | Model architecture | `resnet18` |
| `--dataset` | Dataset name | `cifar10` |
| `--batch-size` | Batch size | `128` |
| `--fault-prob` | Fault probability (%) | `5.0` |
| `--checkpoint` | Path to checkpoint | `None` |
| `--device` | Device (cuda/cpu) | `cuda` |
| `--num-workers` | DataLoader workers | `4` |

---

## Expected Results

### FP16 Mantissa Faults (10-bit mantissa)

| Fault Prob | Expected Accuracy Drop |
|------------|------------------------|
| 0%         | 0% (baseline)          |
| 1%         | ~0.5%                  |
| 5%         | ~2-4%                  |
| 10%        | ~5-8%                  |
| 20%        | ~10-15%                |
| 50%        | ~30-40%                |

### FP16 Exponent Faults (5-bit exponent)

| Fault Prob | Expected Accuracy Drop |
|------------|------------------------|
| 1%         | ~5-10%                 |
| 5%         | ~20-30%                |
| 10%        | ~40-50%                |

**Exponent faults are much more severe!**

---

## Script Details

The script [`evaluate_fp16_faults.py`](evaluate_fp16_faults.py):

1. **Loads model** and converts to FP16
2. **Baseline evaluation** (no faults)
3. **Injects FP fault layers** (mantissa bit-flips)
4. **Faulty evaluation** (with faults)
5. **Prints statistics** (RMSE, cosine similarity, injection rates)
6. **Compares results** (accuracy drop)

---

## Advanced: Custom Script

```python
from utils.fault_injection import FPActivationInjector, FaultInjectionConfig

# Your model (FP16)
model = model.half()

# Configure
config = FaultInjectionConfig(
    enabled=True,
    target_type="activation",
    probability=5.0,
    fp_precision="fp16",
    fp_target_region="mantissa",
    apply_during="eval",
)

# Inject
injector = FPActivationInjector()
model = injector.inject(model, config)

# Evaluate
accuracy = evaluate(model, test_loader)
```

---

## Troubleshooting

### Issue: "Model has no Conv2d/Linear layers"
**Solution**: Check model architecture, adjust `target_layers`

### Issue: "CUDA out of memory"
**Solution**: Reduce `--batch-size`

### Issue: "Accuracy drop is 0%"
**Solution**: Increase `--fault-prob` or check that `fp_precision` matches model dtype

---

## Next Steps

- Compare FP16 vs FP32 fault resilience
- Test different IEEE-754 regions (sign, exponent, mantissa)
- Try fault-aware training to improve resilience
- Analyze per-layer fault impact

---

**Enjoy testing! 🚀**
