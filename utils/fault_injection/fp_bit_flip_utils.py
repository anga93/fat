"""
Floating-point bit-flip utilities for fault injection.
Adapted from MXformats-vs-floatingpoint project.

Provides IEEE-754 bit-level manipulation for FP32 and FP16 formats.
"""

import struct
import random
import torch
from typing import Literal, Optional

Region = Literal["sign", "exponent", "mantissa"]


# === Helpers: float32 <-> 32-bit int && float16 <-> 16-bit int ===

def float32_to_bits(x: float) -> int:
    """
    Interpret a Python float as IEEE-754 float32 and return the 32-bit pattern
    as an unsigned integer.
    """
    # '>f' = big-endian float32, '>I' = big-endian unsigned int32
    return struct.unpack('>I', struct.pack('>f', x))[0]


def bits_to_float32(bits: int) -> float:
    """
    Interpret a 32-bit unsigned integer as an IEEE-754 float32 and return
    a Python float.
    """
    return struct.unpack('>f', struct.pack('>I', bits))[0]


def float16_to_bits(x: float) -> int:
    """
    Interpret a Python float as IEEE-754 float16 and return the 16-bit pattern
    as an unsigned integer.
    """
    # '>e' = big-endian float16, '>H' = big-endian unsigned uint16
    return struct.unpack('>H', struct.pack('>e', x))[0]


def bits_to_float16(bits: int) -> float:
    """
    Interpret a 16-bit unsigned integer as an IEEE-754 float16 and return
    a Python float.
    """
    return struct.unpack('>e', struct.pack('>H', bits & 0xFFFF))[0]


def format_float32_bits(x: float) -> str:
    """
    Return a human-readable string of the float32 bit pattern:
    S | EEEEEEEE | MMMMMMMMMMMMMMMMMMMMMMM
    """
    bits = float32_to_bits(x)
    s = f"{bits:032b}"
    return f"{s[31]} | {s[23:31]} | {s[0:23]}"  # sign | exponent | mantissa


def format_float16_bits(x: float) -> str:
    """
    Return a human-readable string of the float16 bit pattern:
    S | EEEEE | MMMMMMMMMM
    """
    bits = float16_to_bits(x)
    s = f"{bits:016b}"      # 16 bits
    sign = s[0]             # bit 15
    exponent = s[1:6]       # bits 14..10 (5 bits)
    mantissa = s[6:]        # bits 9..0 (10 bits)
    return f"{sign} | {exponent} | {mantissa}"


# === Region definitions ===

# Bits are numbered: 0 = LSB, 31 = MSB
REGIONS_FP32 = {
    "mantissa": (0, 23),   # bits 0-22
    "exponent": (23, 8),   # bits 23-30
    "sign":     (31, 1),   # bit 31
}

# Bits for IEEE-754 binary16 (fp16), numbered 0 = LSB, 15 = MSB
REGIONS_FP16 = {
    "mantissa": (0, 10),   # bits 0-9
    "exponent": (10, 5),   # bits 10-14
    "sign":     (15, 1),   # bit 15
}


# === Bit flip core ===

def flip_bit_in_float32(
    x: float,
    region: Region,
    bit_position: Optional[int] = None,
    verbose: bool = False,
) -> float:
    """
    Flip ONE bit inside the specified region of a float32 number.

    Parameters
    ----------
    x : float
        Input value (interpreted as float32).
    region : {"sign", "exponent", "mantissa"}
        Which field to target.
    bit_position : int or None
        Specific bit index within region (0 = LSB of region).
        If None, chooses randomly.
    verbose : bool
        If True, print bit patterns before and after.

    Returns
    -------
    float
        New float32 value after bit flip.
    """
    if region not in REGIONS_FP32:
        raise ValueError(f"Unknown region '{region}'. Valid: {list(REGIONS_FP32.keys())}")

    start_bit, length = REGIONS_FP32[region]

    # Choose bit position
    if bit_position is None:
        bit_index = random.randrange(length)
    else:
        if not (0 <= bit_position < length):
            raise ValueError(f"bit_position {bit_position} out of range [0, {length-1}] for region '{region}'")
        bit_index = bit_position

    # Compute the global bit position (0 = LSB)
    global_bit_pos = start_bit + bit_index

    # Convert float -> bits
    bits_before = float32_to_bits(x)

    # Create mask and flip
    mask = 1 << global_bit_pos
    bits_after = bits_before ^ mask

    # Convert bits -> float
    y = bits_to_float32(bits_after)

    if verbose:
        print(f"Original value: {x!r}")
        print(f"Original bits: {format_float32_bits(x)}")
        print(f"Region: {region}, bit_index: {bit_index} (global bit {global_bit_pos})")
        print(f"Flipped bits: {format_float32_bits(y)}")
        print(f"New value: {y!r}")

    return y


def flip_bit_in_float16(
    x: float,
    region: Region,
    bit_position: Optional[int] = None,
    verbose: bool = False,
) -> float:
    """
    Flip ONE bit inside the specified region of a float16 number.

    Parameters
    ----------
    x : float
        Input value (interpreted as float16 when encoding).
    region : {"sign", "exponent", "mantissa"}
        Which field to target.
    bit_position : int or None
        Specific bit index within region (0 = LSB of region).
        If None, chooses randomly.
    verbose : bool
        If True, print bit patterns before and after.

    Returns
    -------
    float
        New float16 value after bit flip (returned as Python float).
    """
    if region not in REGIONS_FP16:
        raise ValueError(
            f"Unknown region '{region}'. Valid: {list(REGIONS_FP16.keys())}"
        )

    start_bit, length = REGIONS_FP16[region]

    # Choose bit position
    if bit_position is None:
        bit_index = random.randrange(length)
    else:
        if not (0 <= bit_position < length):
            raise ValueError(f"bit_position {bit_position} out of range [0, {length-1}] for region '{region}'")
        bit_index = bit_position

    # Compute the global bit position (0 = LSB)
    global_bit_pos = start_bit + bit_index

    # Convert float -> bits (fp16 encoding)
    bits_before = float16_to_bits(x)

    # Create mask and flip
    mask = 1 << global_bit_pos
    bits_after = bits_before ^ mask

    # Convert bits -> float16 -> Python float
    y = bits_to_float16(bits_after)

    if verbose:
        print(f"Original value: {x!r}")
        print(f"Original bits: {format_float16_bits(x)}")
        print(
            f"Region: {region}, bit_index: {bit_index} "
            f"(global bit {global_bit_pos})"
        )
        print(f"Flipped bits:  {format_float16_bits(y)}")
        print(f"New value:     {y!r}")

    return y


# === Tensor-level injection ===

def inject_bit_faults_tensor(
    x: torch.Tensor,
    mask: torch.Tensor,
    precision: str = "fp32",
    region: Region = "mantissa",
    bit_position: Optional[int] = None,
    inplace: bool = False,
) -> torch.Tensor:
    """
    Inject random single-bit faults into a float tensor (fp32 or fp16).

    Each element where mask=True gets exactly ONE bit flip in the specified region.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor (dtype=torch.float32 or torch.float16).
    mask : torch.Tensor
        Boolean mask (same shape as x). True = inject fault at this position.
    precision : {"fp32", "fp16"}
        Floating-point format that defines the bit layout.
    region : {"sign", "exponent", "mantissa"}
        Which IEEE-754 region to target.
    bit_position : int or None
        Specific bit within region (None = random for each element).
    inplace : bool
        If True, modify x in place. If False, work on a cloned copy.

    Returns
    -------
    torch.Tensor
        Tensor with injected faults (same shape/device as x).
    """
    if precision not in ("fp32", "fp16"):
        raise ValueError(f"precision must be 'fp32' or 'fp16', got {precision!r}")

    # dtype check
    if precision == "fp32":
        if x.dtype != torch.float32:
            raise ValueError(f"For precision='fp32', tensor must be float32, got {x.dtype}")
        flip_fn = flip_bit_in_float32
    else:  # precision == "fp16"
        if x.dtype != torch.float16:
            raise ValueError(f"For precision='fp16', tensor must be float16, got {x.dtype}")
        flip_fn = flip_bit_in_float16

    # Choose working tensor
    out = x if inplace else x.clone()

    # Get indices where mask is True
    fault_indices = torch.nonzero(mask, as_tuple=False)

    # Apply bit flips
    for idx in fault_indices:
        idx_tuple = tuple(idx.tolist())
        val = out[idx_tuple].item()
        corrupted = flip_fn(val, region=region, bit_position=bit_position, verbose=False)
        out[idx_tuple] = corrupted

    return out


def ser_to_fault_rate(
    R_bit_per_day: float,
    T_days: float,
    bits_per_weight: int = 32,
) -> float:
    """
    Convert per-bit SEU rate in space (upsets/bit/day) into tensor fault_rate.

    Parameters
    ----------
    R_bit_per_day : float
        Single Event Upset rate (bit flips per bit per day).
    T_days : float
        Mission duration in days.
    bits_per_weight : int
        Number of bits per parameter (32 for fp32, 16 for fp16).

    Returns
    -------
    float
        Fault rate (probability that a parameter is corrupted).
    """
    return bits_per_weight * R_bit_per_day * T_days
