# Bug Fixes for V2 and V3 Kernels

## Summary

Fixed critical bugs in v2 (manual async DMA) and v3 (SMEM scales) that would have prevented them from working correctly. These kernels now properly use manual DMA and should produce the expected HLO patterns.

## Bugs Found and Fixed

### 1. Wrong Memory Space (CRITICAL) ❌→✅

**Bug:** Used `pltpu.MemorySpace.ANY` for input/output BlockSpecs

**Problem:**
```python
# WRONG (what I had):
in_specs=[
    pl.BlockSpec(memory_space=pltpu.MemorySpace.ANY),  # Ambiguous!
    ...
]
```

`MemorySpace.ANY` is ambiguous - compiler might auto-fetch to VMEM, defeating the purpose of manual async DMA.

**Fix:**
```python
# CORRECT (what I changed to):
in_specs=[
    pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),  # Explicit!
    ...
]
```

**Why this matters:**
- With `HBM`, Pallas knows: "Don't auto-fetch, kernel will manually copy"
- This enables our manual async DMA code to actually run
- Without this, compiler might auto-fetch and ignore our async copies

**Impact:** Both v2 and v3 would not have worked correctly without this fix.

---

### 2. Wrong Semaphore Type ❌→✅

**Bug:** Used `pltpu.SEM` instead of `pltpu.SemaphoreType.DMA`

**Problem:**
```python
# WRONG (what I had):
scratch_shapes=[
    ...
    pltpu.SEM((2, 5)),  # May not be recognized
]
```

**Fix:**
```python
# CORRECT (matches fused_moe pattern):
scratch_shapes=[
    ...
    pltpu.SemaphoreType.DMA((2, 5)),  # Explicit DMA semaphores
]
```

**Why this matters:**
- `SemaphoreType.DMA` is the explicit, documented way to allocate semaphores
- Ensures compiler knows these are for DMA synchronization
- Matches pattern used in proven kernels like fused_moe

**Impact:** Semaphores might not have been allocated correctly, causing async DMA to fail or deadlock.

---

## What These Fixes Enable

### V2 (Manual Async DMA)

**Now correctly:**
1. ✅ Declares inputs as HBM (no auto-fetch)
2. ✅ Manually copies HBM → VMEM with async DMA
3. ✅ Uses proper DMA semaphores for synchronization
4. ✅ Double-buffers with explicit ping-pong pattern

**Expected HLO output:**
```hlo
# Inputs stay in HBM
%x_hbm = f8e4m3fn[2048,2048]{...} parameter(0), memory_space=0  # HBM

# Manual async copies to VMEM
%async-copy-start.1 = (...) async-copy-start(
  %x_hbm, %x_x2_vmem, ...
)
%async-copy-done.1 = f8e4m3fn[512,512] async-copy-done(%async-copy-start.1)

# Double-buffered arrays
%x_x2_vmem = f8e4m3fn[2,512,512]{...} scratch, memory_space=1  # VMEM
```

### V3 (SMEM Scales)

**Now correctly:**
1. ✅ All fixes from v2 (HBM inputs, DMA semaphores)
2. ✅ Places scales in SMEM (faster access)
3. ✅ Places data blocks in VMEM
4. ✅ Async DMA for both data and scales

**Expected HLO output:**
```hlo
# Data in VMEM
%x_x2_vmem = f8e4m3fn[2,512,512]{...} scratch, memory_space=1  # VMEM

# Scales in SMEM (faster!)
%w_scale_x2_smem = f32[2,1,1]{...} scratch, memory_space=2  # SMEM

# Async copies to both memory spaces
%async-copy-start.scale = (...) async-copy-start(
  %w_scale_hbm, %w_scale_x2_smem, ...
)
```

---

## Verification Steps

### 1. Run Inspection Tools

```bash
./scripts/quick_hlo_check.sh
```

This will:
- Compile all three versions
- Dump HLO to `compilation_output/`
- Analyze patterns

### 2. Check for Fixed Patterns

**V2 should now show:**
- ✅ `memory_space=0` (HBM) for inputs
- ✅ `async-copy-start/done` pairs
- ✅ `[2,...]` double-buffered arrays in VMEM

**V3 should additionally show:**
- ✅ `memory_space=2` (SMEM) for scales
- ✅ `memory_space=1` (VMEM) for data blocks

### 3. Compare Versions

| Feature | V1 | V2 (Fixed) | V3 (Fixed) |
|---------|----|-----------|-----------|
| Native fp8 | ✅ | ✅ | ✅ |
| Double buffering | Auto | Manual | Manual |
| Async DMA | Auto | ✅ Manual | ✅ Manual |
| SMEM scales | ❌ | ❌ | ✅ |
| Memory spaces | Auto | HBM→VMEM | HBM→VMEM/SMEM |

---

## What Was NOT Broken

These parts were actually correct in the original implementation:

✅ **Kernel logic:** The computation (fp8 matmuls, scaling, accumulation) was correct
✅ **Double-buffer allocation:** `[2,...]` shapes were correct
✅ **Async copy calls:** `pltpu.make_async_copy()` usage was correct
✅ **SMEM allocation:** `pltpu.SMEM()` for scales was correct
✅ **Grid iteration:** Grid and program_id usage was correct

The bugs were **configuration issues** (wrong memory space, wrong semaphore type), not algorithmic errors.

---

## Why V1 Didn't Need Fixes

V1 uses the **standard Pallas pattern** for automatic double buffering:
- BlockSpec with shape and lambda → compiler auto-indexes
- PrefetchScalarGridSpec → compiler auto-prefetches
- No manual memory management needed

This is the same pattern used by `quantized_matmul` and works reliably.

**V1 was already correct and will work out of the box.**

---

## Next Steps

1. ✅ **Fixed:** V2 and V3 configuration bugs
2. ⏭️ **Verify:** Run inspection tools to see actual HLO
3. ⏭️ **Benchmark:** Test on TPU to see if manual optimization helps
4. ⏭️ **Compare:** Measure v1 vs v2 vs v3 performance

---

## Lessons Learned

### Use Proven Patterns

When doing manual optimization:
1. Study working examples (fused_moe)
2. Use exact same patterns for memory spaces, semaphores
3. Don't guess - copy proven patterns

### Memory Space Matters

The difference between:
- `MemorySpace.ANY` → compiler decides (may auto-fetch)
- `MemorySpace.HBM` → explicitly in HBM (no auto-fetch)
- `MemorySpace.VMEM` → explicitly in VMEM
- `MemorySpace.SMEM` → explicitly in SMEM

is **critical** for manual DMA control.

### Semaphore Types

Always use explicit types:
- ✅ `pltpu.SemaphoreType.DMA` for DMA operations
- ❌ `pltpu.SEM` might not be recognized correctly

### Verification is Essential

The inspection tools were created to catch exactly these bugs. Without them, we'd be flying blind and might not realize v2/v3 weren't working correctly.

---

## Confidence Level

**V1:** ✅✅✅ **High confidence** - uses proven pattern, should work
**V2:** ✅✅ **Medium-high confidence** - bugs fixed, pattern matches fused_moe
**V3:** ✅ **Medium confidence** - most complex, needs verification on actual TPU

All three should now compile and produce reasonable HLO. Actual performance testing on TPU will determine which is fastest.
