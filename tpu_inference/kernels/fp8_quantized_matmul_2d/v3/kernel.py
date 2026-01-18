# SPDX-License-Identifier: Apache-2.0
"""V3: SMEM for scales + async DMA (aligned blocks only).

This version uses SMEM (Scalar Memory) for scale factors, which is faster than VMEM
for small frequently-accessed data. Combined with async DMA for data transfers.

Key features:
- SMEM for scale storage (faster access than VMEM)
- Async DMA for data blocks
- Double-buffered data in VMEM
- Optimal for small scale tensors

Trade-offs:
- SMEM capacity is limited (~32KB)
- Only beneficial when scales are small
- Aligned blocks only (scales are 1×1 per kernel)
- More complex memory hierarchy management
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tpu_inference.kernels.fp8_quantized_matmul_2d.v3 import util
from tpu_inference.kernels.fp8_quantized_matmul_2d.v3.tuned_block_sizes import (
    TunedValue,
    get_device_vmem_limit,
    get_tuned_block_sizes,
)
from tpu_inference.kernels.fp8_quantized_matmul_2d.v3.util import (
    get_kernel_name,
    next_multiple,
    unfold_args,
)

quantize_tensor_2d = util.quantize_tensor_2d
cdiv = pl.cdiv


def matmul_kernel_2d_smem(
    # HBM inputs
    x_hbm: jax.Array,
    w_q_hbm: jax.Array,
    w_scale_hbm: jax.Array,
    x_abs_max_hbm: jax.Array,
    out_hbm: jax.Array,
    # VMEM scratch (double buffered for data)
    x_x2_vmem: jax.Array,  # (2, quant_block_size, quant_block_size)
    w_q_x2_vmem: jax.Array,  # (2, quant_block_size, quant_block_size)
    out_x2_vmem: jax.Array,  # (2, quant_block_size, quant_block_size)
    acc_vmem: jax.Array,  # (quant_block_size, quant_block_size)
    x_q_scratch: jax.Array,  # (quant_block_size, quant_block_size)
    x_scale_scratch_vmem: jax.Array,  # (1, 1) in VMEM
    # SMEM scratch (for scales - faster access)
    w_scale_x2_smem: jax.Array,  # (2, 1, 1) in SMEM
    x_abs_max_x2_smem: jax.Array,  # (2, 1, 1) in SMEM
    x_scale_scratch_smem: jax.Array,  # (1, 1) in SMEM
    sems: jax.Array,  # (2, 5)
    *,
    x_q_dtype: jnp.dtype,
    quant_block_size: int,
    save_x_q: bool,
):
    """Pallas kernel with SMEM for scales + async DMA.

    V3 OPTIMIZATIONS:
    - SMEM for scale factors (faster than VMEM for small data)
    - Async DMA for data blocks
    - Double-buffered data transfers
    - Reduced VMEM pressure (scales in SMEM)

    Memory hierarchy:
    - Data blocks (x, w_q, out): VMEM (double buffered)
    - Scales (w_scale, x_abs_max, x_scale): SMEM (double buffered)
    - Accumulator: VMEM (single buffer)
    """
    batch_idx, out_idx, in_idx = pl.program_id(0), pl.program_id(1), pl.program_id(2)
    n_in = pl.num_programs(2)

    x_ref_dtype = x_hbm.dtype
    quantize_activation = x_q_dtype != x_ref_dtype

    def get_buffer_id(idx):
        return idx % 2

    def start_fetch(b_idx, o_idx, i_idx):
        """Start async fetch of all data for iteration i_idx."""
        buf_id = get_buffer_id(i_idx)

        # Fetch data blocks to VMEM
        pltpu.make_async_copy(
            src_ref=x_hbm.at[
                pl.ds(b_idx * quant_block_size, quant_block_size),
                pl.ds(i_idx * quant_block_size, quant_block_size),
            ],
            dst_ref=x_x2_vmem.at[buf_id],
            sem=sems.at[buf_id, 0],
        ).start()

        pltpu.make_async_copy(
            src_ref=w_q_hbm.at[
                pl.ds(o_idx * quant_block_size, quant_block_size),
                pl.ds(i_idx * quant_block_size, quant_block_size),
            ],
            dst_ref=w_q_x2_vmem.at[buf_id],
            sem=sems.at[buf_id, 1],
        ).start()

        # Fetch scales to SMEM (faster access)
        pltpu.make_async_copy(
            src_ref=w_scale_hbm.at[o_idx:o_idx+1, i_idx:i_idx+1],
            dst_ref=w_scale_x2_smem.at[buf_id],
            sem=sems.at[buf_id, 2],
        ).start()

        if quantize_activation:
            pltpu.make_async_copy(
                src_ref=x_abs_max_hbm.at[b_idx:b_idx+1, i_idx:i_idx+1],
                dst_ref=x_abs_max_x2_smem.at[buf_id],
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
            src_ref=w_scale_x2_smem.at[buf_id],
            dst_ref=w_scale_x2_smem.at[buf_id],
            sem=sems.at[buf_id, 2],
        ).wait()

        if quantize_activation:
            pltpu.make_async_copy(
                src_ref=x_abs_max_x2_smem.at[buf_id],
                dst_ref=x_abs_max_x2_smem.at[buf_id],
                sem=sems.at[buf_id, 3],
            ).wait()

    def start_store_out(b_idx, o_idx):
        buf_id = get_buffer_id(in_idx)
        pltpu.make_async_copy(
            src_ref=out_x2_vmem.at[buf_id],
            dst_ref=out_hbm.at[
                pl.ds(b_idx * quant_block_size, quant_block_size),
                pl.ds(o_idx * quant_block_size, quant_block_size),
            ],
            sem=sems.at[buf_id, 4],
        ).start()

    def wait_store(i_idx):
        buf_id = get_buffer_id(i_idx)
        pltpu.make_async_copy(
            src_ref=out_x2_vmem.at[buf_id],
            dst_ref=out_x2_vmem.at[buf_id],
            sem=sems.at[buf_id, 4],
        ).wait()

    # Main computation
    is_first_step = in_idx == 0
    is_last_step = in_idx == (n_in - 1)

    # Prefetch
    if is_first_step:
        start_fetch(batch_idx, out_idx, in_idx)
    if not is_last_step:
        start_fetch(batch_idx, out_idx, in_idx + 1)

    # Wait for current data
    wait_fetch(in_idx)
    buf_id = get_buffer_id(in_idx)

    # Quantize activation if needed
    if quantize_activation:
        if out_idx == 0 or not save_x_q:
            # OPTIMIZATION: Read abs_max from SMEM (faster than VMEM)
            x_q_tmp, x_scale_tmp_vmem = util.quantize_array_2d(
                x_x2_vmem[buf_id],
                x_abs_max_x2_smem[buf_id],  # SMEM access
                x_q_dtype,
                quant_block_size,
                quant_block_size,
            )
            # Store scale to SMEM for next access
            x_scale_scratch_smem[...] = x_scale_tmp_vmem

            if save_x_q and out_idx == 0:
                x_q_scratch[...] = x_q_tmp
                x_scale_scratch_vmem[...] = x_scale_tmp_vmem
        else:
            x_q_tmp = x_q_scratch[...]
            if is_last_step:
                x_scale_tmp_vmem = x_scale_scratch_vmem[...]
                x_scale_scratch_smem[...] = x_scale_tmp_vmem

        # Native fp8×fp8 matmul
        acc = jax.lax.dot_general(
            x_q_tmp,
            w_q_x2_vmem[buf_id],
            (((1,), (1,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
    else:
        acc = jax.lax.dot_general(
            x_x2_vmem[buf_id],
            w_q_x2_vmem[buf_id],
            (((1,), (1,)), ((), ())),
            preferred_element_type=jnp.float32,
        )

    # Accumulate
    if not is_first_step:
        acc += acc_vmem[...]

    # Output
    if is_last_step:
        acc = acc.astype(jnp.float32)
        # OPTIMIZATION: Read scale from SMEM (faster access)
        acc *= w_scale_x2_smem[buf_id, 0, 0]
        if quantize_activation:
            acc *= x_scale_scratch_smem[0, 0]

        out_x2_vmem[buf_id, ...] = acc.astype(x_ref_dtype)

        if in_idx > 0:
            wait_store(in_idx - 1)
        start_store_out(batch_idx, out_idx)
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
    x: jax.Array,
    w_q: jax.Array,
    w_scale: jax.Array,
    w_zp: jax.Array | None = None,
    x_q_dtype: jnp.dtype | None = None,
    quant_block_size: int = 128,
    *,
    tuned_value: TunedValue | None = None,
) -> jax.Array:
    """V3: 2D fp8 quantized matmul with SMEM for scales.

    PERFORMANCE-CRITICAL OPTIMIZATIONS:
    - **SMEM for scales**: Faster access than VMEM for small data
    - **Native fp8×fp8 matmul**: Hardware MXU acceleration
    - **Async DMA**: Overlaps compute with memory transfers
    - **Reduced VMEM pressure**: Scales moved to SMEM

    Benefits of SMEM:
    - ~2-3× faster access than VMEM for small data
    - Reduces VMEM contention
    - Perfect for scales (just 1×1 per aligned block)

    NOTE: Currently only supports aligned blocks.
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

    # [Same preprocessing as v2...]
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

    # V3 currently only supports aligned blocks
    if (batch_block_size != quant_block_size or
        out_block_size != quant_block_size or
        in_block_size != quant_block_size):
        raise NotImplementedError(
            "V3 (SMEM) currently only supports aligned blocks. "
            f"Got batch_block={batch_block_size}, out_block={out_block_size}, "
            f"in_block={in_block_size}, quant_block={quant_block_size}"
        )

    # [Same padding as v2...]
    padded_n_batch = next_multiple(orig_n_batch, batch_block_size)
    if orig_n_batch < padded_n_batch:
        x = jnp.pad(x, ((0, padded_n_batch - orig_n_batch), (0, 0)))
        if quantize_activation:
            x_abs_max = jnp.pad(
                x_abs_max,
                ((0, (padded_n_batch // quant_block_size) - x_abs_max.shape[0]), (0, 0)),
            )

    padded_n_out = next_multiple(orig_n_out, out_block_size)
    if orig_n_out < padded_n_out:
        w_q = jnp.pad(w_q, ((0, padded_n_out - orig_n_out), (0, 0)))
        w_scale = jnp.pad(
            w_scale,
            ((0, (padded_n_out // quant_block_size) - w_scale.shape[0]), (0, 0)),
        )

    padded_n_in = next_multiple(orig_n_in, in_block_size)
    if orig_n_in < padded_n_in:
        x = jnp.pad(x, ((0, 0), (0, padded_n_in - orig_n_in)))
        w_q = jnp.pad(w_q, ((0, 0), (0, padded_n_in - orig_n_in)))
        if quantize_activation:
            x_abs_max = jnp.pad(
                x_abs_max,
                ((0, 0), (0, (padded_n_in // quant_block_size) - x_abs_max.shape[1])),
            )
        w_scale = jnp.pad(
            w_scale,
            ((0, 0), (0, (padded_n_in // quant_block_size) - w_scale.shape[1])),
        )

    if w_scale.dtype != jnp.float32:
        w_scale = w_scale.astype(jnp.float32)

    n_batch = padded_n_batch // batch_block_size
    n_out = padded_n_out // out_block_size
    n_in = padded_n_in // in_block_size

    save_x_q = quantize_activation and n_in == 1 and n_out > 1

    # V3: SMEM for scales, VMEM for data
    kernel = pl.pallas_call(
        functools.partial(
            matmul_kernel_2d_smem,
            x_q_dtype=x_q_dtype,
            quant_block_size=quant_block_size,
            save_x_q=save_x_q,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),  # x_hbm
                pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),  # w_q_hbm
                pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),  # w_scale_hbm
                pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),  # x_abs_max_hbm
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
            scratch_shapes=[
                # VMEM for data blocks (double buffered)
                pltpu.VMEM((2, quant_block_size, quant_block_size), x.dtype),  # x_x2
                pltpu.VMEM((2, quant_block_size, quant_block_size), w_q.dtype),  # w_q_x2
                pltpu.VMEM((2, quant_block_size, quant_block_size), x.dtype),  # out_x2
                pltpu.VMEM((quant_block_size, quant_block_size), jnp.float32),  # acc
                pltpu.VMEM((quant_block_size, quant_block_size), x_q_dtype) if save_x_q else None,  # x_q
                pltpu.VMEM((1, 1), jnp.float32) if save_x_q else None,  # x_scale_vmem
                # SMEM for scales (faster access, double buffered)
                pltpu.SMEM((2, 1, 1), jnp.float32),  # w_scale_x2_smem
                pltpu.SMEM((2, 1, 1), jnp.float32),  # x_abs_max_x2_smem
                pltpu.SMEM((1, 1), jnp.float32),  # x_scale_smem
                # Semaphores for async DMA synchronization
                pltpu.SemaphoreType.DMA((2, 5)),
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
