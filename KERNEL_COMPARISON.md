# Pallas Kernel Comparison: fp8_quantized_matmul_2d vs Existing Kernels

## Overview
This document compares the 2D fp8 quantized matmul kernel with existing Pallas kernels in the codebase:
- **quantized_matmul** (1D/per-channel quantization)
- **fused_moe/v1** (sub-channel quantization with MoE)
- **ragged_paged_attention/v3** (attention with paging and double buffering)

---

## 1. Matmul Computation Strategy

### quantized_matmul (1D)
```python
# Lines 85-97: Single dot_general for entire block
acc = jax.lax.dot_general(
    x_q_tmp,
    w_q_ref[...],
    (((1,), (1,)), ((), ())),
    preferred_element_type=acc_dtype,
)
# Then scale AFTER accumulation (line 103)
acc *= w_scale_ref[...]  # 1D scale: [1, out_block_size]
```
**Pattern**: Compute full matmul → scale once at the end
**Works for**: Per-channel/per-token quantization (1D scales)

### fp8_quantized_matmul_2d (Ours)
```python
# Lines 123-152: Sub-block iteration with per-block scaling
for batch_qblock_id in range(n_batch_quant_blocks):
    for out_qblock_id in range(n_out_quant_blocks):
        partial_acc = jnp.zeros(...)
        for in_qblock_id in range(n_in_quant_blocks):
            # Extract sub-blocks
            x_block = x_input[pl.ds(...), pl.ds(...)]
            w_block = w_q_ref[pl.ds(...), pl.ds(...)]

            # Compute sub-matmul
            sub_result = jnp.dot(x_block, w_block.T, ...)

            # Apply 2D scale BEFORE accumulation
            combined_scale = x_scale[batch_qblock_id, in_qblock_id] * \
                            w_scale[out_qblock_id, in_qblock_id]
            sub_result *= combined_scale
            partial_acc += sub_result
```
**Pattern**: Iterate sub-blocks → compute sub-matmul → scale immediately → accumulate
**Works for**: 2D block-wise quantization (scales vary by both row and column blocks)

### fused_moe/v1 (Sub-channel)
```python
# Lines 813-845: Similar sub-block pattern
for bd1c_id in range(cdiv(bd1, bd1c)):
    for bfc_id in range(cdiv(bf, bfc)):
        acc1 = jnp.dot(t, w1, preferred_element_type=jnp.float32)

        # Apply sub-channel scale (line 834-845)
        w1_scale_slices = (
            p_id,
            (bd1c_id * bd1c_per_t_packing) // subc_quant_w1_sz,
            pl.ds(0, 1),
            pl.ds(bfc_id * bfc, bfc),
        )
        w1_scale = jnp.broadcast_to(w1_scale_vmem[*w1_scale_slices], acc1.shape)
        acc1 *= w1_scale  # Scale before accumulation
```
**Pattern**: Sub-block iteration → compute → scale → accumulate (same as ours!)
**Works for**: Sub-channel quantization (1D along one axis, but requires sub-blocking)

---

## 2. Static vs Dynamic Block Sizes

### quantized_matmul (1D)
```python
# Block sizes NOT passed to kernel function
def matmul_kernel(x_ref, w_q_ref, ...):
    batch_block_size, in_block_size = x_ref.shape  # Runtime shape
    out_block_size = w_q_ref.shape[0]
```
**Issue**: Block sizes are inferred from ref shapes at runtime
**Works**: For simple 1D case without sub-iteration

### fp8_quantized_matmul_2d (Ours)
```python
# Lines 40-42: Block sizes as static parameters
def matmul_kernel_2d(
    x_ref, w_q_ref, ...,
    *,
    quant_block_size: int,
    batch_block_size: int,  # ← Static!
    out_block_size: int,    # ← Static!
    in_block_size: int,     # ← Static!
):
    # Lines 80-82: Use for compile-time loop bounds
    n_batch_quant_blocks = batch_block_size // quant_block_size
    n_out_quant_blocks = out_block_size // quant_block_size
    n_in_quant_blocks = in_block_size // quant_block_size
```
**Advantage**: Compiler knows loop bounds at compile time → better optimization
**Pattern**: Matches fused_moe's approach

### fused_moe/v1
```python
# Lines 232-246: Block sizes as static parameters
def _fused_ep_moe_kernel(..., *, bt, bf, bd1, bd2, btc, bfc, bd1c, bd2c):
    # Line 265-272: Extensive static assertions
    assert bd1 % bd1c == 0
    assert bd2 % bd2c == 0
    assert bf % bfc == 0
    # ... more assertions ...
```
**Best practice**: Static block sizes + compile-time assertions

---

## 3. TPU Constraint Enforcement

### quantized_matmul (1D)
```python
# Lines 161-164: Only minor dim alignment
x_abs_max = jnp.expand_dims(x_abs_max, axis=0)  # [1, bs]
# Comment: "Pallas requires minormost dim to be multiple of sublane size 128"
```
**Limited**: Only enforces 128 alignment for minor dimension

### fp8_quantized_matmul_2d (Ours)
```python
# Lines 60-66: Comprehensive compile-time assertions
assert batch_block_size % quant_block_size == 0
assert out_block_size % quant_block_size == 0
assert in_block_size % quant_block_size == 0
assert quant_block_size % 128 == 0  # Must be multiple of 128 for TPU
assert batch_block_size % 8 == 0    # (8x128) divisibility
assert out_block_size % 128 == 0
assert in_block_size % 128 == 0
```
**Complete**: Enforces all TPU constraints at compile time

### fused_moe/v1
```python
# Lines 281-297: Detailed sub-channel quantization constraints
if subc_quant_w1_sz is not None:
    if subc_quant_w1_sz < hidden_size:
        assert subc_quant_w1_sz % 256 == 0
        assert bd1c_per_t_packing == subc_quant_w1_sz
        assert bd1 % subc_quant_w1_sz == 0
        assert bd1_per_t_packing % subc_quant_w1_sz == 0
        # ... more ...
```
**Excellent**: Very comprehensive constraint checking

---

## 4. Memory Access Patterns

### quantized_matmul (1D)
```python
# Lines 241-249: Simple BlockSpec with lambda indexing
in_specs=[
    pl.BlockSpec((batch_block_size, in_block_size),
                lambda b, o, i: (b, i)),  # x
    pl.BlockSpec((out_block_size, in_block_size),
                lambda b, o, i: (o, i)),  # w_q
    pl.BlockSpec((1, out_block_size),
                lambda b, o, i: (0, o)),  # w_scale (1D)
]
```
**Simple**: Lambda-based indexing for straightforward access patterns

### fp8_quantized_matmul_2d (Ours)
```python
# Lines 424-437: 2D scale blocks with lambda indexing
in_specs=[
    pl.BlockSpec((batch_block_size, in_block_size),
                lambda b, o, i: (b, i)),
    pl.BlockSpec((out_block_size, in_block_size),
                lambda b, o, i: (o, i)),
    pl.BlockSpec((n_w_blocks_m, n_w_blocks_n),  # 2D scales!
                lambda b, o, i: (o * n_w_blocks_m, i * n_w_blocks_n)),
]
```
**2D Scales**: Properly maps 2D quantization blocks to grid

### ragged_paged_attention/v3
```python
# Lines 448-524: Complex async DMA with semaphores
def _fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx, *, wait=False):
    sem = sems.at[0, bkv_sem_idx]
    # Lines 492-505: Pipelined DMA with pl.ds()
    for i in range(bkv_p):
        sz = jnp.clip(kv_left_frm_cache - i * page_size, 0, page_size)
        _async_copy(
            cache_hbm_ref.at[pl.ds(page_indices_ref[page_idx] * page_size, sz)],
            vmem_ref.at[pl.ds(i * page_size, sz)],
            sem,
            wait=False,
        )
```
**Advanced**: Asynchronous DMA with double buffering and dynamic sizing

---

## 5. Double Buffering & Async Copies

### quantized_matmul (1D)
```python
# NO async copies or explicit double buffering
# Uses scratch_shapes for accumulation only:
scratch_shapes=[
    pltpu.VMEM((batch_block_size, out_block_size), acc_dtype) if save_acc else None,
]
```
**Basic**: Simple VMEM scratch for accumulation

### fp8_quantized_matmul_2d (Ours)
```python
# Lines 444-456: Scratch buffers but NO async copies
scratch_shapes=[
    pltpu.VMEM((batch_block_size, out_block_size), acc_dtype) if save_acc else None,
    pltpu.VMEM((batch_block_size, in_block_size), x_q_dtype) if save_x_q else None,
    pltpu.VMEM((n_x_blocks_m, n_x_blocks_n), jnp.float32) if save_x_q else None,
]
```
**Same as 1D**: No async copies - **MISSING OPTIMIZATION**

### fused_moe/v1
```python
# Lines 212-223: Explicit double buffering with semaphores
b_w1_x2_vmem,   # <bw_sem_id> (2, ...)  ← _x2 = double buffer
b_w3_x2_vmem,   # <bw_sem_id> (2, ...)
b_w2_x2_vmem,   # <bw_sem_id> (2, ...)
# ...
local_sems,  # (2, 5): semaphores for coordination

# Lines 700-750: Async copy pattern
def wait_fetch_bw1(local_e_id, bw1_sem_id, bf_id, bd1_id):
    pltpu.make_async_copy(
        src_ref=b_w1_x2_vmem.at[bw1_sem_id],
        dst_ref=b_w1_x2_vmem.at[bw1_sem_id],
        sem=local_sems.at[bw1_sem_id, 1],
    ).wait()
```
**Production**: Full double buffering with semaphore-controlled async DMA

### ragged_paged_attention/v3
```python
# Lines 256-259: Double buffered refs
bkv_x2_ref,  # [2, bkv_sz, ...]  ← Double buffered
bq_x2_ref,   # [2, actual_num_kv_heads, bq_sz, ...]
bo_x2_ref,   # [2, ...]
sems,        # [4, 2]  ← 4 types x 2 buffers

# Lines 438-446: Async copy helper
def _async_copy(src, dst, sem, wait):
    cp = pltpu.make_async_copy(src, dst, sem)
    if wait:
        cp.wait()
    else:
        cp.start()
```
**Advanced**: Full pipelined DMA with prefetching

---

## 6. Scale Indexing Pattern

### quantized_matmul (1D)
```python
# Line 103: Simple broadcast multiplication
acc *= w_scale_ref[...]  # Shape: [1, out_block_size]
# x_scale is [batch_block_size, 1], broadcasts naturally
```
**Simple**: Direct multiplication, no indexing needed

### fp8_quantized_matmul_2d (Ours)
```python
# Lines 144-149: 2D indexing
w_scale_val = w_scale_ref[out_qblock_id, in_qblock_id]
if quantize_activation:
    x_scale_val = x_scale_tmp[batch_qblock_id, in_qblock_id]
    combined_scale = x_scale_val * w_scale_val
```
**2D Indexing**: Direct array indexing into 2D scale array

### fused_moe/v1
```python
# Lines 835-845: Sub-channel scale indexing with division
w1_scale_slices = (
    p_id,
    (bd1c_id * bd1c_per_t_packing) // subc_quant_w1_sz,  # ← Key pattern!
    pl.ds(0, 1),
    pl.ds(bfc_id * bfc, bfc),
)
w1_scale = jnp.broadcast_to(w1_scale_vmem[*w1_scale_slices], acc1.shape)
acc1 *= w1_scale
```
**Pattern**: `(block_id * block_size) // quant_size` for scale index
**Our version matches this conceptually!**

---

## 7. Dynamic Slicing Usage

### quantized_matmul (1D)
```python
# NO pl.ds() usage - entire blocks always accessed
x_ref[...]  # Full block access
w_q_ref[...]
```
**None**: Not needed for 1D case

### fp8_quantized_matmul_2d (Ours)
```python
# Lines 131-138: pl.ds() for sub-block extraction
x_block = x_input[
    pl.ds(batch_qblock_id * quant_block_size, quant_block_size),
    pl.ds(in_qblock_id * quant_block_size, quant_block_size),
]
w_block = w_q_ref[
    pl.ds(out_qblock_id * quant_block_size, quant_block_size),
    pl.ds(in_qblock_id * quant_block_size, quant_block_size),
]
```
**Correct**: Uses pl.ds(offset, size) for dynamic slicing with static size
**Matches fused_moe pattern**

### fused_moe/v1
```python
# Lines 815-827: pl.ds() everywhere
t_b32 = t_b32_vmem[
    pl.ds(btc_id * btc, btc),
    pl.ds(bd1c_id * bd1c_per_t_packing, bd1c_per_t_packing),
]
# Lines 865-866
acc_slices = (
    pl.ds(btc_id * btc, btc),
    pl.ds(bfc_id * bfc, bfc)
)
```
**Heavy usage**: pl.ds() for all sub-block accesses

---

## Summary Scorecard

| Feature | quantized_matmul (1D) | fp8_quantized_matmul_2d (Ours) | fused_moe/v1 | ragged_paged_attention/v3 |
|---------|----------------------|-------------------------------|--------------|--------------------------|
| **Sub-block iteration** | ❌ No | ✅ Yes | ✅ Yes | ✅ Yes |
| **Scales before accumulation** | ❌ After | ✅ Before | ✅ Before | N/A |
| **Static block sizes** | ❌ Runtime | ✅ Static params | ✅ Static params | ✅ Static params |
| **Compile-time assertions** | ⚠️ Basic | ✅ Comprehensive | ✅ Excellent | ✅ Excellent |
| **TPU constraint checks** | ⚠️ Limited | ✅ Complete | ✅ Complete | ✅ Complete |
| **pl.ds() usage** | ❌ No | ✅ Yes | ✅ Yes | ✅ Yes |
| **Double buffering** | ❌ No | ❌ No | ✅ Yes | ✅ Yes |
| **Async DMA copies** | ❌ No | ❌ No | ✅ Yes | ✅ Yes |
| **Semaphore coordination** | ❌ No | ❌ No | ✅ Yes | ✅ Yes |
| **Complexity** | Simple | Medium | High | Very High |

---

## Key Insights

### ✅ What We Did Right:
1. **Sub-block iteration pattern**: Matches fused_moe's approach for sub-channel quantization
2. **Static block sizes**: All sizes passed as compile-time constants
3. **Scales before accumulation**: Correct approach for 2D quantization
4. **pl.ds() for slicing**: Proper dynamic slicing with static sizes
5. **Comprehensive assertions**: All TPU constraints checked at compile time
6. **2D scale indexing**: Correctly maps quantization blocks to scales

### ⚠️ What We're Missing (Compared to Production Kernels):
1. **No double buffering**: Could pipeline computation and DMA
2. **No async copies**: Missing `pltpu.make_async_copy()` for prefetching
3. **No semaphores**: No `pltpu.SemaphoreType.DMA` for coordination
4. **No data packing**: Missing bitcasting tricks for int8/fp8 packing
5. **No dynamic loop bounds**: Could use `lax.fori_loop` for dynamic sizing

### 🎯 Our Kernel Position:
- **More sophisticated than**: quantized_matmul (1D)
- **Similar complexity to**: Early version of fused_moe's sub-channel logic
- **Less sophisticated than**: Production fused_moe and ragged_paged_attention

---

## Recommendations for Future Improvements

### Priority 1: Double Buffering (High Impact)
Add double buffering for weight blocks following fused_moe pattern:
```python
# Instead of single w_q_ref, use:
w_q_x2_ref,  # [2, out_block_size, in_block_size]
w_scale_x2_ref,  # [2, n_w_blocks_m, n_w_blocks_n]
sems,  # [2] for coordination

# Prefetch next weight block while computing current
```

### Priority 2: Async DMA (High Impact)
Add `pltpu.make_async_copy()` to pipeline data movement:
```python
def start_fetch_w_block(out_idx, in_idx, sem_idx):
    pltpu.make_async_copy(
        src_ref=w_q_hbm[pl.ds(out_idx * out_block_size, out_block_size),
                        pl.ds(in_idx * in_block_size, in_block_size)],
        dst_ref=w_q_x2_vmem.at[sem_idx],
        sem=sems.at[sem_idx],
    ).start()  # Non-blocking
```

### Priority 3: Dynamic Loop Bounds (Medium Impact)
Use `lax.fori_loop` for better handling of partial blocks:
```python
def process_block(block_id, carry):
    # ... sub-block computation ...
    return carry

lax.fori_loop(0, n_batch_quant_blocks, process_block, init_carry)
```

### Priority 4: Bitcasting Optimizations (Medium Impact)
Add packing/unpacking for better memory efficiency (if needed for other dtypes)

---

## Conclusion

Our `fp8_quantized_matmul_2d` kernel successfully implements the **core algorithmic pattern** for 2D block-wise quantization by following the sub-block iteration approach from `fused_moe`. The kernel is **correct** and **well-structured** with proper:
- Static block sizes for compiler optimization
- Sub-block iteration for correct 2D scaling
- Comprehensive TPU constraint enforcement
- Proper use of `pl.ds()` for dynamic slicing

However, it's missing the **performance optimizations** found in production kernels like double buffering and async DMA. These would be natural next steps to improve throughput, especially for larger models where data movement becomes the bottleneck.

The kernel represents a **solid foundation** - algorithmically correct with good compile-time optimization, but with room to add the advanced pipelining techniques used in production TPU kernels.
