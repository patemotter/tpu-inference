# SPDX-License-Identifier: Apache-2.0
"""Tuned block sizes for 2D fp8 quantized matmul kernel on Ironwood TPU."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TunedValue:
    """Tuned block size configuration for the kernel.

    Attributes:
        batch_block_size: Block size for the batch dimension
        out_block_size: Block size for the output features dimension
        in_block_size: Block size for the input features dimension
        quant_block_size: Block size for 2D quantization (128, 256, or 512)
    """
    batch_block_size: int
    out_block_size: int
    in_block_size: int
    quant_block_size: int


# Ironwood TPU configuration
# - 64MB VMEM
# - 256x256 MXU with native fp8 support
# - Blocks must be divisible by (8x128) for last two dimensions
IRONWOOD_VMEM_LIMIT = 64 * 1024 * 1024  # 64 MB


def get_device_vmem_limit():
    """Return the VMEM limit for Ironwood TPU."""
    return IRONWOOD_VMEM_LIMIT


# Tuned block sizes database
# Key: (n_batch, n_out, n_in, x_q_dtype, w_q_dtype, quant_block_size)
# Value: TunedValue(batch_block_size, out_block_size, in_block_size, quant_block_size)
#
# These are starting configurations optimized for:
# - Ironwood TPU (64MB VMEM, 256x256 MXU)
# - fp8_e4m3fn quantization
# - 2D block quantization with blocks of 128x128, 256x256, or 512x512
# - Divisibility by (8x128) for last two dimensions

TUNED_BLOCK_SIZES = {
    # Small matmuls with 128x128 quantization blocks
    # Format: (batch, out, in, x_dtype, w_dtype, quant_block) -> (batch_block, out_block, in_block, quant_block)

    # 128x128 quantization blocks
    (1024, 1024, 1024, "float8_e4m3fn", "float8_e4m3fn", 128): TunedValue(1024, 1024, 1024, 128),
    (2048, 2048, 2048, "float8_e4m3fn", "float8_e4m3fn", 128): TunedValue(1024, 1024, 2048, 128),
    (4096, 4096, 4096, "float8_e4m3fn", "float8_e4m3fn", 128): TunedValue(1024, 2048, 4096, 128),
    (1024, 2048, 1024, "float8_e4m3fn", "float8_e4m3fn", 128): TunedValue(1024, 1024, 1024, 128),
    (1024, 4096, 2048, "float8_e4m3fn", "float8_e4m3fn", 128): TunedValue(1024, 2048, 2048, 128),
    (2048, 8192, 2048, "float8_e4m3fn", "float8_e4m3fn", 128): TunedValue(1024, 4096, 2048, 128),

    # 256x256 quantization blocks
    (1024, 1024, 1024, "float8_e4m3fn", "float8_e4m3fn", 256): TunedValue(1024, 1024, 1024, 256),
    (2048, 2048, 2048, "float8_e4m3fn", "float8_e4m3fn", 256): TunedValue(2048, 2048, 2048, 256),
    (4096, 4096, 4096, "float8_e4m3fn", "float8_e4m3fn", 256): TunedValue(2048, 2048, 4096, 256),
    (1024, 2048, 1024, "float8_e4m3fn", "float8_e4m3fn", 256): TunedValue(1024, 2048, 1024, 256),
    (1024, 4096, 2048, "float8_e4m3fn", "float8_e4m3fn", 256): TunedValue(1024, 2048, 2048, 256),
    (2048, 8192, 2048, "float8_e4m3fn", "float8_e4m3fn", 256): TunedValue(2048, 4096, 2048, 256),
    (4096, 16384, 4096, "float8_e4m3fn", "float8_e4m3fn", 256): TunedValue(2048, 4096, 4096, 256),

    # 512x512 quantization blocks
    (1024, 1024, 1024, "float8_e4m3fn", "float8_e4m3fn", 512): TunedValue(1024, 1024, 1024, 512),
    (2048, 2048, 2048, "float8_e4m3fn", "float8_e4m3fn", 512): TunedValue(2048, 2048, 2048, 512),
    (4096, 4096, 4096, "float8_e4m3fn", "float8_e4m3fn", 512): TunedValue(4096, 4096, 4096, 512),
    (8192, 8192, 8192, "float8_e4m3fn", "float8_e4m3fn", 512): TunedValue(4096, 4096, 8192, 512),
    (1024, 2048, 1024, "float8_e4m3fn", "float8_e4m3fn", 512): TunedValue(1024, 2048, 1024, 512),
    (2048, 4096, 2048, "float8_e4m3fn", "float8_e4m3fn", 512): TunedValue(2048, 4096, 2048, 512),
    (4096, 8192, 4096, "float8_e4m3fn", "float8_e4m3fn", 512): TunedValue(4096, 4096, 4096, 512),

    # Mixed precision: bfloat16 activation with fp8 weights
    # 128x128 quantization blocks
    (1024, 1024, 1024, "bfloat16", "float8_e4m3fn", 128): TunedValue(1024, 1024, 1024, 128),
    (2048, 2048, 2048, "bfloat16", "float8_e4m3fn", 128): TunedValue(1024, 1024, 2048, 128),
    (4096, 4096, 4096, "bfloat16", "float8_e4m3fn", 128): TunedValue(1024, 2048, 4096, 128),

    # 256x256 quantization blocks
    (1024, 1024, 1024, "bfloat16", "float8_e4m3fn", 256): TunedValue(1024, 1024, 1024, 256),
    (2048, 2048, 2048, "bfloat16", "float8_e4m3fn", 256): TunedValue(2048, 2048, 2048, 256),
    (4096, 4096, 4096, "bfloat16", "float8_e4m3fn", 256): TunedValue(2048, 2048, 4096, 256),

    # 512x512 quantization blocks
    (1024, 1024, 1024, "bfloat16", "float8_e4m3fn", 512): TunedValue(1024, 1024, 1024, 512),
    (2048, 2048, 2048, "bfloat16", "float8_e4m3fn", 512): TunedValue(2048, 2048, 2048, 512),
    (4096, 4096, 4096, "bfloat16", "float8_e4m3fn", 512): TunedValue(4096, 4096, 4096, 512),
}


def get_tuned_block_sizes(
    n_batch: int,
    n_out: int,
    n_in: int,
    x_q_dtype: str,
    w_q_dtype: str,
    quant_block_size: int,
) -> TunedValue:
    """Get tuned block sizes for the given configuration.

    Args:
        n_batch: Batch size
        n_out: Output feature dimension
        n_in: Input feature dimension
        x_q_dtype: Activation quantization dtype name
        w_q_dtype: Weight quantization dtype name
        quant_block_size: 2D quantization block size (128, 256, or 512)

    Returns:
        TunedValue with optimized block sizes
    """
    key = (n_batch, n_out, n_in, x_q_dtype, w_q_dtype, quant_block_size)

    # Try exact match first
    if key in TUNED_BLOCK_SIZES:
        return TUNED_BLOCK_SIZES[key]

    # Fall back to default heuristic based on quantization block size
    # These defaults ensure:
    # 1. Divisibility by (8x128) for last two dimensions
    # 2. Reasonable VMEM usage for Ironwood (64MB)
    # 3. Good MXU utilization (256x256)

    if quant_block_size == 128:
        # For 128x128 quantization, use smaller kernel blocks
        batch_block_size = min(n_batch, 1024)
        out_block_size = min(n_out, 1024)
        in_block_size = min(n_in, 2048)
    elif quant_block_size == 256:
        # For 256x256 quantization, use medium kernel blocks
        batch_block_size = min(n_batch, 2048)
        out_block_size = min(n_out, 2048)
        in_block_size = min(n_in, 4096)
    elif quant_block_size == 512:
        # For 512x512 quantization, use larger kernel blocks
        batch_block_size = min(n_batch, 4096)
        out_block_size = min(n_out, 4096)
        in_block_size = min(n_in, 8192)
    else:
        raise ValueError(
            f"Unsupported quantization block size: {quant_block_size}. "
            "Supported values are: 128, 256, 512"
        )

    # Ensure divisibility by quant_block_size
    batch_block_size = max(quant_block_size, (batch_block_size // quant_block_size) * quant_block_size)
    out_block_size = max(quant_block_size, (out_block_size // quant_block_size) * quant_block_size)
    in_block_size = max(quant_block_size, (in_block_size // quant_block_size) * quant_block_size)

    return TunedValue(
        batch_block_size=batch_block_size,
        out_block_size=out_block_size,
        in_block_size=in_block_size,
        quant_block_size=quant_block_size,
    )
