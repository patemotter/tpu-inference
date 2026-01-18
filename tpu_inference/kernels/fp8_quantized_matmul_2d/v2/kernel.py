# SPDX-License-Identifier: Apache-2.0
"""V2: Manual async DMA with explicit semaphores (aligned blocks only).

This version uses explicit async DMA operations with manual semaphore synchronization,
similar to the fused_moe kernel. This gives more control over prefetch timing.

Key features:
- Explicit pltpu.make_async_copy() for all transfers
- Manual semaphore-based synchronization
- Double-buffered VMEM arrays (x2 pattern)
- Fine-grained control over prefetch timing

Trade-offs:
- More complex code
- Manual buffer management
- Potentially better performance if compiler auto-prefetch is suboptimal
- Currently only supports aligned blocks (kernel_block == quant_block)
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tpu_inference.kernels.fp8_quantized_matmul_2d.v2 import util
from tpu_inference.kernels.fp8_quantized_matmul_2d.v2.tuned_block_sizes import (
    TunedValue,
    get_device_vmem_limit,
    get_tuned_block_sizes,
)
from tpu_inference.kernels.fp8_quantized_matmul_2d.v2.util import (
    get_kernel_name,
    next_multiple,
    unfold_args,
)

quantize_tensor_2d = util.quantize_tensor_2d
cdiv = pl.cdiv


def matmul_kernel_2d_async_dma(
    # HBM inputs
    x_hbm: jax.Array,  # (padded_n_batch, padded_n_in)
    w_q_hbm: jax.Array,  # (padded_n_out, padded_n_in)
    w_scale_hbm: jax.Array,  # (n_out_blocks, n_in_blocks)
    x_abs_max_hbm: jax.Array,  # (n_batch_blocks, n_in_blocks)
    out_hbm: jax.Array,  # (padded_n_batch, padded_n_out)
    # VMEM scratch (double buffered)
    x_x2_vmem: jax.Array,  # (2, quant_block_size, quant_block_size)
    w_q_x2_vmem: jax.Array,  # (2, quant_block_size, quant_block_size)
    w_scale_x2_vmem: jax.Array,  # (2, 1, 1)
    x_abs_max_x2_vmem: jax.Array,  # (2, 1, 1)
    out_x2_vmem: jax.Array,  # (2, quant_block_size, quant_block_size)
    acc_vmem: jax.Array,  # (quant_block_size, quant_block_size) - not double buffered
    x_q_scratch: jax.Array,  # (quant_block_size, quant_block_size)
    x_scale_scratch: jax.Array,  # (1, 1)
    sems: jax.Array,  # (2, 5) - semaphores for synchronization
    *,
    x_q_dtype: jnp.dtype,
    quant_block_size: int,
    save_x_q: bool,
):
    """Pallas kernel with EXPLICIT async DMA and semaphores.

    V2 OPTIMIZATIONS:
    - Manual async copy with pltpu.make_async_copy()
    - Explicit semaphore synchronization
    - Double-buffered inputs (x2 pattern)
    - Fine-grained control over prefetch timing

    Semaphore layout (sems[buffer_id, sem_type]):
    - sem 0: x transfer
    - sem 1: w_q transfer
    - sem 2: w_scale transfer
    - sem 3: x_abs_max transfer
    - sem 4: out transfer
    """
    # Grid indices
    batch_idx, out_idx, in_idx = pl.program_id(0), pl.program_id(1), pl.program_id(2)
    n_in = pl.num_programs(2)

    x_ref_dtype = x_hbm.dtype
    quantize_activation = x_q_dtype != x_ref_dtype

    # Semaphore helpers
    def get_buffer_id(idx):
        """Get buffer ID (0 or 1) for double buffering."""
        return idx % 2

    def start_fetch_x(b_idx, o_idx, i_idx):
        """Start async fetch of activation block."""
        buf_id = get_buffer_id(i_idx)
        pltpu.make_async_copy(
            src_ref=x_hbm.at[
                pl.ds(b_idx * quant_block_size, quant_block_size),
                pl.ds(i_idx * quant_block_size, quant_block_size),
            ],
            dst_ref=x_x2_vmem.at[buf_id],
            sem=sems.at[buf_id, 0],
        ).start()

    def start_fetch_w(b_idx, o_idx, i_idx):
        """Start async fetch of weight block."""
        buf_id = get_buffer_id(i_idx)
        pltpu.make_async_copy(
            src_ref=w_q_hbm.at[
                pl.ds(o_idx * quant_block_size, quant_block_size),
                pl.ds(i_idx * quant_block_size, quant_block_size),
            ],
            dst_ref=w_q_x2_vmem.at[buf_id],
            sem=sems.at[buf_id, 1],
        ).start()

    def start_fetch_scales(b_idx, o_idx, i_idx):
        """Start async fetch of scales."""
        buf_id = get_buffer_id(i_idx)
        pltpu.make_async_copy(
            src_ref=w_scale_hbm.at[o_idx:o_idx+1, i_idx:i_idx+1],
            dst_ref=w_scale_x2_vmem.at[buf_id],
            sem=sems.at[buf_id, 2],
        ).start()
        if quantize_activation:
            pltpu.make_async_copy(
                src_ref=x_abs_max_hbm.at[b_idx:b_idx+1, i_idx:i_idx+1],
                dst_ref=x_abs_max_x2_vmem.at[buf_id],
                sem=sems.at[buf_id, 3],
            ).start()

    def wait_fetch(i_idx):
        """Wait for async fetches to complete."""
        buf_id = get_buffer_id(i_idx)
        pltpu.make_async_copy(
            src_ref=x_x2_vmem.at[buf_id],
            dst_ref=x_x2_vmem.at[buf_id],
            sem=sems.at[buf_id, 0],
        ).wait()
        pltpu.make_async_copy(
            src_ref=w_q_x2_vmem.at[buf_id],
            dst_ref=w_q_x2_vmem.at[buf_id],
            sem=sems.at[buf_id, 1],
        ).wait()
        pltpu.make_async_copy(
            src_ref=w_scale_x2_vmem.at[buf_id],
            dst_ref=w_scale_x2_vmem.at[buf_id],
            sem=sems.at[buf_id, 2],
        ).wait()
        if quantize_activation:
            pltpu.make_async_copy(
                src_ref=x_abs_max_x2_vmem.at[buf_id],
                dst_ref=x_abs_max_x2_vmem.at[buf_id],
                sem=sems.at[buf_id, 3],
            ).wait()

    def start_store_out(b_idx, o_idx):
        """Start async store of output."""
        buf_id = get_buffer_id(in_idx)  # Use current in_idx
        pltpu.make_async_copy(
            src_ref=out_x2_vmem.at[buf_id],
            dst_ref=out_hbm.at[
                pl.ds(b_idx * quant_block_size, quant_block_size),
                pl.ds(o_idx * quant_block_size, quant_block_size),
            ],
            sem=sems.at[buf_id, 4],
        ).start()

    def wait_store(i_idx):
        """Wait for output store to complete."""
        buf_id = get_buffer_id(i_idx)
        pltpu.make_async_copy(
            src_ref=out_x2_vmem.at[buf_id],
            dst_ref=out_x2_vmem.at[buf_id],
            sem=sems.at[buf_id, 4],
        ).wait()

    # Main computation loop
    is_first_step = in_idx == 0
    is_last_step = in_idx == (n_in - 1)

    # Prefetch first iteration
    if is_first_step:
        start_fetch_x(batch_idx, out_idx, in_idx)
        start_fetch_w(batch_idx, out_idx, in_idx)
        start_fetch_scales(batch_idx, out_idx, in_idx)

    # Prefetch next iteration (if not last)
    if not is_last_step:
        start_fetch_x(batch_idx, out_idx, in_idx + 1)
        start_fetch_w(batch_idx, out_idx, in_idx + 1)
        start_fetch_scales(batch_idx, out_idx, in_idx + 1)

    # Wait for current iteration data
    wait_fetch(in_idx)

    buf_id = get_buffer_id(in_idx)

    # Quantize activation if needed
    if quantize_activation:
        if out_idx == 0 or not save_x_q:  # Quantize
            x_q_tmp, x_scale_tmp = util.quantize_array_2d(
                x_x2_vmem[buf_id],
                x_abs_max_x2_vmem[buf_id],
                x_q_dtype,
                quant_block_size,
                quant_block_size,
            )
            if save_x_q and out_idx == 0:
                x_q_scratch[...] = x_q_tmp
                x_scale_scratch[...] = x_scale_tmp
        else:  # Reuse quantized
            x_q_tmp = x_q_scratch[...]
            if is_last_step:
                x_scale_tmp = x_scale_scratch[...]

        # Native fp8×fp8 matmul
        acc = jax.lax.dot_general(
            x_q_tmp,
            w_q_x2_vmem[buf_id],
            (((1,), (1,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
    else:
        # bf16 × fp8
        acc = jax.lax.dot_general(
            x_x2_vmem[buf_id],
            w_q_x2_vmem[buf_id],
            (((1,), (1,)), ((), ())),
            preferred_element_type=jnp.float32,
        )

    # Accumulate across in dimension
    if not is_first_step:
        acc += acc_vmem[...]

    # Output on last step
    if is_last_step:
        # Apply scales
        acc = acc.astype(jnp.float32)
        acc *= w_scale_x2_vmem[buf_id, 0, 0]
        if quantize_activation:
            acc *= x_scale_tmp[0, 0]

        # Store to output buffer (will be written to HBM async)
        out_x2_vmem[buf_id, ...] = acc.astype(x_ref_dtype)

        # Wait for previous output store (if any)
        if in_idx > 0:
            wait_store(in_idx - 1)

        # Start storing current output
        start_store_out(batch_idx, out_idx)

        # Wait for current output to finish
        wait_store(in_idx)
    else:
        acc_vmem[...] = acc


@functools.partial(
    jax.jit,
    static_argnames=[
        "x_q_dtype",
        "quant_block_size",
        "tuned_value",
    ],
)
def fp8_quantized_matmul_2d_kernel(
    x: jax.Array,  # [bs, n_in]
    w_q: jax.Array,  # [n_out, n_in]
    w_scale: jax.Array,  # [n_out // quant_block_size, n_in // quant_block_size]
    w_zp: jax.Array | None = None,
    x_q_dtype: jnp.dtype | None = None,
    quant_block_size: int = 128,
    *,
    tuned_value: TunedValue | None = None,
) -> jax.Array:
    """V2: 2D fp8 quantized matmul with explicit async DMA.

    PERFORMANCE-CRITICAL OPTIMIZATIONS:
    - **Native fp8×fp8 matmul**: Leverages hardware fp8 MXU
    - **EXPLICIT ASYNC DMA**: Manual control over prefetch timing with semaphores
    - **Double buffered VMEM**: Overlaps compute with memory transfers
    - **Aligned blocks**: kernel_block == quant_block for simplicity

    NOTE: Currently only supports aligned blocks. Large blocks support coming soon.
    """

    if w_zp is not None:
        raise NotImplementedError("zero_point is not supported.")

    if quant_block_size not in [128, 256, 512]:
        raise ValueError(
            f"quant_block_size must be 128, 256, or 512, got {quant_block_size}"
        )

    if x_q_dtype is None:
        x_q_dtype = x.dtype
    quantize_activation = x_q_dtype != x.dtype

    orig_n_batch, orig_n_in = x.shape
    orig_n_out, _ = w_q.shape

    # Compute 2D abs max for activation blocks if quantizing
    if quantize_activation:
        padded_n_batch_for_quant = next_multiple(orig_n_batch, quant_block_size)
        padded_n_in_for_quant = next_multiple(orig_n_in, quant_block_size)

        x_for_abs_max = x
        if orig_n_batch < padded_n_batch_for_quant or orig_n_in < padded_n_in_for_quant:
            x_for_abs_max = jnp.pad(
                x,
                (
                    (0, padded_n_batch_for_quant - orig_n_batch),
                    (0, padded_n_in_for_quant - orig_n_in),
                ),
            )

        n_quant_blocks_m = padded_n_batch_for_quant // quant_block_size
        n_quant_blocks_n = padded_n_in_for_quant // quant_block_size

        x_reshaped = x_for_abs_max.reshape(
            n_quant_blocks_m,
            quant_block_size,
            n_quant_blocks_n,
            quant_block_size,
        )
        x_abs_max = jnp.max(jnp.abs(x_reshaped), axis=(1, 3))
    else:
        x_abs_max = jnp.zeros((1, 1), dtype=jnp.float32)

    if tuned_value is None:
        tuned_value = get_tuned_block_sizes(
            n_batch=orig_n_batch,
            n_out=orig_n_out,
            n_in=orig_n_in,
            x_q_dtype=jnp.dtype(x_q_dtype).name,
            w_q_dtype=jnp.dtype(w_q.dtype).name,
            quant_block_size=quant_block_size,
        )

    batch_block_size = tuned_value.batch_block_size
    out_block_size = tuned_value.out_block_size
    in_block_size = tuned_value.in_block_size

    # V2 currently only supports aligned blocks
    if (batch_block_size != quant_block_size or
        out_block_size != quant_block_size or
        in_block_size != quant_block_size):
        raise NotImplementedError(
            "V2 (async DMA) currently only supports aligned blocks. "
            f"Got batch_block={batch_block_size}, out_block={out_block_size}, "
            f"in_block={in_block_size}, quant_block={quant_block_size}"
        )

    # Pad inputs
    padded_n_batch = next_multiple(orig_n_batch, batch_block_size)
    if orig_n_batch < padded_n_batch:
        x = jnp.pad(x, ((0, padded_n_batch - orig_n_batch), (0, 0)))
        if quantize_activation:
            x_abs_max = jnp.pad(
                x_abs_max,
                (
                    (0, (padded_n_batch // quant_block_size) - x_abs_max.shape[0]),
                    (0, 0),
                ),
            )

    padded_n_out = next_multiple(orig_n_out, out_block_size)
    if orig_n_out < padded_n_out:
        w_q = jnp.pad(w_q, ((0, padded_n_out - orig_n_out), (0, 0)))
        w_scale = jnp.pad(
            w_scale,
            (
                (0, (padded_n_out // quant_block_size) - w_scale.shape[0]),
                (0, 0),
            ),
        )

    padded_n_in = next_multiple(orig_n_in, in_block_size)
    if orig_n_in < padded_n_in:
        x = jnp.pad(x, ((0, 0), (0, padded_n_in - orig_n_in)))
        w_q = jnp.pad(w_q, ((0, 0), (0, padded_n_in - orig_n_in)))
        if quantize_activation:
            x_abs_max = jnp.pad(
                x_abs_max,
                (
                    (0, 0),
                    (0, (padded_n_in // quant_block_size) - x_abs_max.shape[1]),
                ),
            )
        w_scale = jnp.pad(
            w_scale,
            (
                (0, 0),
                (0, (padded_n_in // quant_block_size) - w_scale.shape[1]),
            ),
        )

    if w_scale.dtype != jnp.float32:
        w_scale = w_scale.astype(jnp.float32)

    n_batch = padded_n_batch // batch_block_size
    n_out = padded_n_out // out_block_size
    n_in = padded_n_in // in_block_size

    save_x_q = quantize_activation and n_in == 1 and n_out > 1

    # V2: Use explicit BlockSpec with HBM inputs/outputs
    # No automatic prefetching - we handle it manually
    kernel = pl.pallas_call(
        functools.partial(
            matmul_kernel_2d_async_dma,
            x_q_dtype=x_q_dtype,
            quant_block_size=quant_block_size,
            save_x_q=save_x_q,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                # HBM inputs (no automatic prefetch - we do manual async DMA)
                pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),  # x_hbm
                pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),  # w_q_hbm
                pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),  # w_scale_hbm
                pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),  # x_abs_max_hbm
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),  # out_hbm
            scratch_shapes=[
                # Double-buffered VMEM
                pltpu.VMEM((2, quant_block_size, quant_block_size), x.dtype),  # x_x2
                pltpu.VMEM((2, quant_block_size, quant_block_size), w_q.dtype),  # w_q_x2
                pltpu.VMEM((2, 1, 1), jnp.float32),  # w_scale_x2
                pltpu.VMEM((2, 1, 1), jnp.float32),  # x_abs_max_x2
                pltpu.VMEM((2, quant_block_size, quant_block_size), x.dtype),  # out_x2
                # Single-buffered scratch
                pltpu.VMEM((quant_block_size, quant_block_size), jnp.float32),  # acc
                pltpu.VMEM((quant_block_size, quant_block_size), x_q_dtype) if save_x_q else None,  # x_q
                pltpu.VMEM((1, 1), jnp.float32) if save_x_q else None,  # x_scale
                # Semaphores for async DMA synchronization
                pltpu.SemaphoreType.DMA((2, 5)),  # sems[buffer_id, sem_type]
            ],
            grid=(n_batch, n_out, n_in),
        ),
        out_shape=jax.ShapeDtypeStruct((padded_n_batch, padded_n_out), x.dtype),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary", "arbitrary"),
        ),
    )

    kernel_name = get_kernel_name(tuned_value)
    with jax.named_scope(kernel_name):
        out = kernel(x, w_q, w_scale, x_abs_max)

    return out[:orig_n_batch, :orig_n_out]
