# SPDX-License-Identifier: Apache-2.0
"""Utility functions for 2D fp8 quantized matmul kernel."""
from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax._src import dtypes

from tpu_inference.kernels.fp8_quantized_matmul_2d.v1.tuned_block_sizes import TunedValue


def unfold_args(
    conditions: tuple[jax.Array | bool, ...],
    fn_conditions: tuple[bool, ...],
    fn: Callable[..., Any],
):
    """Minimize run-time branching of fn by converting jnp.bool to python bool."""
    if conditions:
        arg = conditions[0]
        if isinstance(arg, bool):
            unfold_args(conditions[1:], fn_conditions + (arg,), fn)
        else:
            # Use shape check instead of size to avoid TracerBoolConversionError
            assert arg.dtype == jnp.bool_
            assert arg.shape == ()
            jax.lax.cond(
                arg,
                lambda: unfold_args(conditions[1:], fn_conditions + (True,), fn),
                lambda: unfold_args(conditions[1:], fn_conditions + (False,), fn),
            )
    else:
        fn(*fn_conditions)


def quantize_tensor_2d(
    x: jax.Array,
    dtype: jnp.dtype,
    block_size_m: int,
    block_size_n: int,
):
    """2D block-wise quantization for tensors.

    Args:
        x: Input tensor of shape [M, N]
        dtype: Target quantization dtype (e.g., jnp.float8_e4m3fn)
        block_size_m: Block size along the M dimension
        block_size_n: Block size along the N dimension

    Returns:
        x_q: Quantized tensor of shape [M, N]
        scale: Scale tensor of shape [M // block_size_m, N // block_size_n]
    """
    M, N = x.shape

    # Ensure dimensions are divisible by block sizes
    assert M % block_size_m == 0, f"M ({M}) must be divisible by block_size_m ({block_size_m})"
    assert N % block_size_n == 0, f"N ({N}) must be divisible by block_size_n ({block_size_n})"

    n_blocks_m = M // block_size_m
    n_blocks_n = N // block_size_n

    # Get dtype info
    if jnp.issubdtype(dtype, jnp.integer):
        dtype_info = jnp.iinfo(dtype)
        max_val = int(dtype_info.max)
        min_val = int(dtype_info.min)
    else:
        dtype_info = jnp.finfo(dtype)
        max_val = float(dtype_info.max)
        min_val = float(dtype_info.min)

    # Reshape to expose blocks: [n_blocks_m, block_size_m, n_blocks_n, block_size_n]
    x_reshaped = x.reshape(n_blocks_m, block_size_m, n_blocks_n, block_size_n)

    # Compute max absolute value per block: [n_blocks_m, n_blocks_n]
    x_abs_max = jnp.max(jnp.abs(x_reshaped), axis=(1, 3))

    # Compute scale: [n_blocks_m, n_blocks_n]
    scale = x_abs_max / max_val

    # Expand scale for broadcasting: [n_blocks_m, 1, n_blocks_n, 1]
    scale_expanded = scale[:, jnp.newaxis, :, jnp.newaxis]

    # Quantize
    x_q_reshaped = jnp.clip(x_reshaped / scale_expanded, min_val, max_val).astype(dtype)

    # Reshape back to original shape
    x_q = x_q_reshaped.reshape(M, N)

    return x_q, scale.astype(jnp.float32)


def quantize_array_2d(
    x: jax.Array,  # [batch_block_size, in_block_size]
    x_abs_max: jax.Array,  # [n_quant_blocks_m, n_quant_blocks_n]
    quant_dtype: jnp.dtype,
    quant_block_size_m: int,
    quant_block_size_n: int,
):
    """Quantize array with 2D block-wise quantization inside kernel.

    Args:
        x: Input array of shape [batch_block_size, in_block_size]
        x_abs_max: Pre-computed abs max per block [n_quant_blocks_m, n_quant_blocks_n]
        quant_dtype: Target quantization dtype
        quant_block_size_m: Quantization block size along M dimension
        quant_block_size_n: Quantization block size along N dimension

    Returns:
        x_q: Quantized array
        scale: Scale array of shape [n_quant_blocks_m, n_quant_blocks_n]
    """
    is_float = jnp.issubdtype(quant_dtype, jnp.floating)
    dtype_info = jnp.finfo(quant_dtype) if is_float else jnp.iinfo(quant_dtype)
    dtype_max = float(dtype_info.max)

    # Compute scale from abs max
    scale = x_abs_max / dtype_max  # [n_quant_blocks_m, n_quant_blocks_n]

    batch_block_size, in_block_size = x.shape
    n_quant_blocks_m = batch_block_size // quant_block_size_m
    n_quant_blocks_n = in_block_size // quant_block_size_n

    # Reshape to expose quantization blocks
    x_reshaped = x.reshape(
        n_quant_blocks_m, quant_block_size_m,
        n_quant_blocks_n, quant_block_size_n
    )

    # Expand scale for broadcasting: [n_quant_blocks_m, 1, n_quant_blocks_n, 1]
    scale_expanded = scale[:, jnp.newaxis, :, jnp.newaxis]

    # Quantize
    x_q_reshaped = (x_reshaped / scale_expanded).astype(quant_dtype)
    x_q = x_q_reshaped.reshape(batch_block_size, in_block_size)

    return x_q, scale.astype(jnp.float32)


def next_multiple(x, multiple):
    """Round up to next multiple."""
    return ((x + multiple - 1) // multiple) * multiple


def get_kernel_name(tuned_value: TunedValue):
    """Generate kernel name from tuned values."""
    batch_block_size = tuned_value.batch_block_size
    out_block_size = tuned_value.out_block_size
    in_block_size = tuned_value.in_block_size
    quant_block_size = tuned_value.quant_block_size
    return (
        f"fp8_quantized_matmul_2d_kernel_{batch_block_size}_{out_block_size}_"
        f"{in_block_size}_{quant_block_size}"
    )


def xla_quantized_matmul_2d(
    x: jax.Array,  # [bs, n_in]
    w_q: jax.Array,  # [n_out, n_in]
    w_scale: jax.Array,  # [n_out // quant_block_size, n_in // quant_block_size]
    x_quantize: bool,
    quant_block_size: int,
) -> jax.Array:
    """Reference (pure JAX) implementation of 2D quantized matmul.

    Args:
        x: Activation array [bs, n_in]
        w_q: Weight quantized array [n_out, n_in]
        w_scale: Weight 2D quantization scale [n_out // quant_block_size, n_in // quant_block_size]
        x_quantize: Whether to quantize activations
        quant_block_size: Size of 2D quantization blocks

    Returns:
        Output of the quantized matmul [bs, n_out]
    """
    bs, n_in = x.shape
    n_out, _ = w_q.shape

    # For reference implementation, we'll do a simple block-wise dequantize and matmul
    # This is not optimized but serves as a correctness reference

    n_blocks_out = n_out // quant_block_size
    n_blocks_in = n_in // quant_block_size

    # Dequantize weights block-wise
    w_dequant = jnp.zeros((n_out, n_in), dtype=jnp.float32)
    for i in range(n_blocks_out):
        for j in range(n_blocks_in):
            block_slice_out = slice(i * quant_block_size, (i + 1) * quant_block_size)
            block_slice_in = slice(j * quant_block_size, (j + 1) * quant_block_size)
            w_block = w_q[block_slice_out, block_slice_in].astype(jnp.float32)
            w_dequant = w_dequant.at[block_slice_out, block_slice_in].set(
                w_block * w_scale[i, j]
            )

    if x_quantize:
        # Quantize x with 2D blocks
        x_q, x_scale = quantize_tensor_2d(x, w_q.dtype, quant_block_size, quant_block_size)

        # Dequantize x block-wise
        n_blocks_bs = bs // quant_block_size
        x_dequant = jnp.zeros((bs, n_in), dtype=jnp.float32)
        for i in range(n_blocks_bs):
            for j in range(n_blocks_in):
                block_slice_bs = slice(i * quant_block_size, (i + 1) * quant_block_size)
                block_slice_in = slice(j * quant_block_size, (j + 1) * quant_block_size)
                x_block = x_q[block_slice_bs, block_slice_in].astype(jnp.float32)
                x_dequant = x_dequant.at[block_slice_bs, block_slice_in].set(
                    x_block * x_scale[i, j]
                )
        x_input = x_dequant
    else:
        x_input = x

    # Perform matmul
    out = jax.lax.dot_general(
        x_input,
        w_dequant,
        dimension_numbers=(((1,), (1,)), ((), ())),
        preferred_element_type=jnp.float32,
    )

    return out.astype(x.dtype)


def get_vmem_limit(
    n_batch: int,
    n_out: int,
    n_in: int,
    batch_block_size: int,
    out_block_size: int,
    in_block_size: int,
    quant_block_size: int,
    x_dtype: jnp.dtype,
    x_q_dtype: jnp.dtype,
    w_q_dtype: jnp.dtype,
    scale_dtype: jnp.dtype,
    out_dtype: jnp.dtype,
    acc_dtype: jnp.dtype,
    save_acc: bool,
    save_x_q: bool,
    upper_limit_bytes: int,
):
    """Calculate VMEM limit for the 2D quantized kernel."""

    def get_bits(dtype):
        if hasattr(dtypes, "bit_width"):
            return dtypes.bit_width(dtype)
        else:
            return dtypes.itemsize_bits(dtype)

    # Calculate in/out VMEM size
    x_size = batch_block_size * in_block_size * get_bits(x_dtype)

    # For 2D quantization, abs_max is per block
    n_quant_blocks_m = batch_block_size // quant_block_size
    n_quant_blocks_n = in_block_size // quant_block_size
    x_abs_max_size = n_quant_blocks_m * n_quant_blocks_n * get_bits(scale_dtype)

    w_q_size = out_block_size * in_block_size * get_bits(w_q_dtype)

    # Weight scale is also 2D
    n_w_blocks_m = out_block_size // quant_block_size
    n_w_blocks_n = in_block_size // quant_block_size
    w_scale_size = n_w_blocks_m * n_w_blocks_n * get_bits(scale_dtype)

    out_size = batch_block_size * out_block_size * get_bits(out_dtype)

    vmem_in_out = x_size + x_abs_max_size + w_q_size + w_scale_size + out_size
    vmem_in_out *= 2  # Account for compute and vreg spills

    # CRITICAL PERFORMANCE: Account for double buffering
    # Double buffering is ESSENTIAL for maximizing TPU performance by overlapping
    # compute with memory transfers. With PrefetchScalarGridSpec, the compiler
    # automatically double buffers inputs when there are multiple grid iterations.
    # We allocate extra VMEM here to enable this:
    vmem_in_out += x_size if (n_batch > 1 or n_in > 1) else 0  # DB activations
    vmem_in_out += x_abs_max_size if (n_batch > 1) else 0  # DB act abs_max
    vmem_in_out += w_q_size if (n_out > 1 or n_in > 1) else 0  # DB weights
    vmem_in_out += w_scale_size if (n_out > 1) else 0  # DB weight scales
    vmem_in_out += out_size if (n_batch > 1 or n_out > 1) else 0  # DB outputs

    # Calculate scratch VMEM size
    acc_size = batch_block_size * out_block_size * get_bits(acc_dtype)
    x_q_size = batch_block_size * in_block_size * get_bits(x_q_dtype)
    x_scale_size = n_quant_blocks_m * n_quant_blocks_n * get_bits(scale_dtype)

    vmem_scratch = acc_size if save_acc else 0
    vmem_scratch += x_q_size + x_scale_size if save_x_q else 0
    vmem_scratch *= 2  # Account for compute and vreg spills

    # Add in/out and scratch VMEM size
    vmem_used = vmem_in_out + vmem_scratch
    vmem_used_bytes = vmem_used // 8  # Convert bits to bytes

    # Specify upper limit
    vmem_limit_bytes = min(vmem_used_bytes, upper_limit_bytes)

    return vmem_limit_bytes


def validate_inputs(
    x: jax.Array,
    w_q: jax.Array,
    w_scale: jax.Array,
    x_abs_max: jax.Array,
    x_q_dtype: jnp.dtype,
    batch_block_size: int,
    out_block_size: int,
    in_block_size: int,
    quant_block_size: int,
):
    """Verify inputs invoking the kernel."""

    if x.dtype != x_q_dtype:
        # If the input is quantized, then it should be the same subdtype as w_q
        if jnp.issubdtype(x_q_dtype, jnp.integer) != jnp.issubdtype(
            w_q.dtype, jnp.integer
        ):
            raise ValueError(
                f"{x_q_dtype=} and {w_q.dtype=} must be the same int or float type."
            )

    # Verify input shapes
    if x.shape[1] != w_q.shape[1]:
        raise ValueError(f"{x.shape[1]=} must be equal to {w_q.shape[1]=}")

    # For 2D quantization, scale shape is 2D
    n_quant_blocks_m = x.shape[0] // quant_block_size
    n_quant_blocks_n = x.shape[1] // quant_block_size
    expected_x_abs_max_shape = (n_quant_blocks_m, n_quant_blocks_n)
    if x_abs_max.shape != expected_x_abs_max_shape:
        raise ValueError(
            f"{x_abs_max.shape=} must be equal to {expected_x_abs_max_shape}"
        )

    n_w_blocks_m = w_q.shape[0] // quant_block_size
    n_w_blocks_n = w_q.shape[1] // quant_block_size
    expected_w_scale_shape = (n_w_blocks_m, n_w_blocks_n)
    if w_scale.shape != expected_w_scale_shape:
        raise ValueError(
            f"{w_scale.shape=} must be equal to {expected_w_scale_shape}"
        )

    if x.shape[0] % batch_block_size != 0:
        raise ValueError(
            f"{x.shape[0]=} must be a multiple of {batch_block_size=}"
        )
    if w_q.shape[0] % out_block_size != 0:
        raise ValueError(
            f"{w_q.shape[0]=} must be a multiple of {out_block_size=}"
        )
    if x.shape[1] % in_block_size != 0:
        raise ValueError(
            f"{x.shape[1]=} must be a multiple of {in_block_size=}"
        )

    # Verify block sizes are divisible by quantization block size
    if batch_block_size % quant_block_size != 0:
        raise ValueError(
            f"{batch_block_size=} must be a multiple of {quant_block_size=}"
        )
    if out_block_size % quant_block_size != 0:
        raise ValueError(
            f"{out_block_size=} must be a multiple of {quant_block_size=}"
        )
    if in_block_size % quant_block_size != 0:
        raise ValueError(
            f"{in_block_size=} must be a multiple of {quant_block_size=}"
        )
