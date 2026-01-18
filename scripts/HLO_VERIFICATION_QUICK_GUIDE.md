# Quick HLO Verification Guide for Bug Fixes

This guide shows exactly what to look for in HLO to verify the v2/v3 bug fixes are working.

## Quick Verification (With JAX Installed)

```bash
# Generate HLO
python scripts/inspect_fp8_2d_kernel_compilation.py

# Run automated verification
./scripts/verify_bugfixes.sh

# Or manual deep analysis
python scripts/analyze_hlo.py compilation_output/*_hlo.txt
```

---

## Manual Verification Patterns

### V2 (Manual Async DMA) - What to Look For

#### 1. HBM Input Declaration ✅

**Look for:**
```hlo
%x_hbm = f8e4m3fn[2048,2048]{1,0} parameter(0), metadata={...}, memory_space=0
%w_q_hbm = f8e4m3fn[4096,2048]{1,0} parameter(1), metadata={...}, memory_space=0
```

**Key pattern:** `memory_space=0` (HBM)

**What it means:**
- ✅ Inputs stay in HBM
- ✅ No automatic prefetching
- ✅ Kernel will manually copy with async DMA

**Red flag:**
```hlo
%x = f8e4m3fn[2048,2048]{1,0} parameter(0)  # No memory_space specified!
```
This means compiler auto-fetched, breaking manual DMA.

---

#### 2. Async DMA Operations ✅

**Look for:**
```hlo
%async-copy-start.1 = (f8e4m3fn[512,512]{1,0}, u32[], ...) async-copy-start(
  f8e4m3fn[2048,2048]{1,0} %x_hbm,
  ...
), memory_space=1

%async-copy-done.1 = f8e4m3fn[512,512]{1,0} async-copy-done(
  (f8e4m3fn[512,512]{1,0}, u32[], ...) %async-copy-start.1
)
```

**Count them:**
```bash
grep -c "async-copy-start" compilation_output/v2_manual_async_dma_hlo.txt
grep -c "async-copy-done" compilation_output/v2_manual_async_dma_hlo.txt
```

**Expected:**
- Should have equal numbers (balanced)
- Typically 10-20+ pairs for our kernel
- Each pair represents one async DMA transfer

**Red flag:**
- Zero async operations → Manual DMA not happening
- Imbalanced (different counts) → Potential deadlock

---

#### 3. Double-Buffered Arrays ✅

**Look for:**
```hlo
%x_x2_vmem = f8e4m3fn[2,512,512]{2,1,0} scratch, memory_space=1
%w_q_x2_vmem = f8e4m3fn[2,512,512]{2,1,0} scratch, memory_space=1
%out_x2_vmem = bf16[2,512,512]{2,1,0} scratch, memory_space=1
```

**Key pattern:** `[2,...]` shape (leading dimension = 2)

**What it means:**
- ✅ Ping-pong buffers allocated
- ✅ Buffer 0 and buffer 1 for double buffering
- ✅ While computing with buffer 0, prefetch to buffer 1

**How to check:**
```bash
grep "\[2," compilation_output/v2_manual_async_dma_hlo.txt
```

Should see multiple arrays with leading dimension of 2.

---

#### 4. Buffer Indexing ✅

**Look for:**
```hlo
%buffer_id_0 = s32[] constant(0)
%buffer_id_1 = s32[] constant(1)

%dynamic-slice.1 = f8e4m3fn[512,512]{1,0} dynamic-slice(
  f8e4m3fn[2,512,512]{2,1,0} %x_x2_vmem,
  s32[] %buffer_id_0,
  ...
)

%dynamic-slice.2 = f8e4m3fn[512,512]{1,0} dynamic-slice(
  f8e4m3fn[2,512,512]{2,1,0} %x_x2_vmem,
  s32[] %buffer_id_1,
  ...
)
```

**What it means:**
- ✅ Explicit buffer selection (0 or 1)
- ✅ Ping-pong pattern in action

---

### V3 (SMEM Scales) - Additional Patterns

Everything from V2, PLUS:

#### 5. SMEM Allocation ✅

**Look for:**
```hlo
%w_scale_x2_smem = f32[2,1,1]{2,1,0} scratch, memory_space=2
%x_abs_max_x2_smem = f32[2,1,1]{2,1,0} scratch, memory_space=2
%x_scale_smem = f32[1,1]{1,0} scratch, memory_space=2
```

**Key pattern:** `memory_space=2` (SMEM)

**What it means:**
- ✅ Scales placed in fast SMEM
- ✅ ~2-3× faster access than VMEM
- ✅ V3 optimization working

**How to check:**
```bash
grep "memory_space=2" compilation_output/v3_smem_scales_hlo.txt
```

Should find scales (small arrays) in SMEM.

---

#### 6. VMEM for Data (not scales) ✅

**Look for:**
```hlo
%x_x2_vmem = f8e4m3fn[2,512,512]{2,1,0} scratch, memory_space=1
%w_q_x2_vmem = f8e4m3fn[2,512,512]{2,1,0} scratch, memory_space=1
```

**Key pattern:** `memory_space=1` (VMEM) for large data

**What it means:**
- ✅ Large data blocks in VMEM (appropriate)
- ✅ Small scales in SMEM (fast)
- ✅ Proper memory hierarchy usage

---

#### 7. Async DMA to SMEM ✅

**Look for:**
```hlo
%async-copy-start.scale = (...) async-copy-start(
  f32[1,1]{1,0} %w_scale_hbm,
  ...
), memory_space=2  # Destination is SMEM
```

**What it means:**
- ✅ Async transfers HBM → SMEM
- ✅ Prefetching scales to fast memory

---

## Common Issues and What They Mean

### Issue: No `memory_space=0` in parameters

**Symptom:**
```hlo
%x = f8e4m3fn[2048,2048]{1,0} parameter(0)  # No memory space!
```

**Problem:** Inputs not explicitly in HBM

**Likely cause:** BlockSpec using `MemorySpace.ANY` instead of `MemorySpace.HBM`

**Impact:** Compiler may auto-fetch, bypassing manual DMA

---

### Issue: No async operations

**Symptom:**
```bash
$ grep -c "async-copy" compilation_output/v2_manual_async_dma_hlo.txt
0
```

**Problem:** No async DMA happening

**Likely causes:**
1. Wrong memory space (inputs auto-fetched)
2. Async copy calls not in kernel
3. Compiler optimized them away

**Impact:** No double buffering, performance loss

---

### Issue: No `[2,...]` arrays

**Symptom:**
```bash
$ grep "\[2," compilation_output/v2_manual_async_dma_hlo.txt
# No matches
```

**Problem:** No double buffering

**Likely cause:** Scratch shapes don't have leading dim of 2

**Impact:** No ping-pong buffers, can't overlap compute/memory

---

### Issue: No SMEM (v3 only)

**Symptom:**
```bash
$ grep "memory_space=2" compilation_output/v3_smem_scales_hlo.txt
# No matches
```

**Problem:** V3 optimization not working

**Likely causes:**
1. SMEM allocation syntax wrong
2. Scales placed in VMEM instead
3. TPU doesn't support SMEM (unlikely)

**Impact:** Slower scale access, v3 = v2 (no benefit)

---

## Automated Checks

### Run Verification Script

```bash
./scripts/verify_bugfixes.sh
```

This automatically checks all patterns and gives a PASS/FAIL verdict.

### Manual Grep Patterns

```bash
# V2 checks
echo "HBM inputs:" && grep -c "parameter.*memory_space=0" v2_*.txt
echo "Async ops:" && grep -c "async-copy-start" v2_*.txt
echo "Double buf:" && grep -c "\[2," v2_*.txt

# V3 checks (all v2 checks plus)
echo "SMEM usage:" && grep -c "memory_space=2" v3_*.txt
echo "VMEM usage:" && grep -c "memory_space=1" v3_*.txt
```

---

## Expected Results

### V2 Success Criteria

- ✅ `memory_space=0` for parameters (5+ occurrences)
- ✅ `async-copy-start` (10+ occurrences)
- ✅ `async-copy-done` (equal to starts)
- ✅ `[2,...]` shaped arrays (3+ occurrences)

### V3 Success Criteria

- ✅ All V2 criteria met
- ✅ `memory_space=2` for scales (3+ occurrences)
- ✅ `memory_space=1` for data (5+ occurrences)
- ✅ Async copies to both VMEM and SMEM

---

## Next Steps After Verification

1. ✅ **HLO looks good** → Proceed to TPU testing
2. ✅ **Get LLO** → Run on TPU to see actual assembly
3. ✅ **Benchmark** → Compare v1 vs v2 vs v3 performance
4. ❌ **HLO looks bad** → Debug kernel code, check syntax

---

## Quick Reference

| Pattern | Meaning | File to Check |
|---------|---------|---------------|
| `memory_space=0` | HBM | v2, v3 |
| `memory_space=1` | VMEM | v2, v3 |
| `memory_space=2` | SMEM | v3 only |
| `async-copy-start` | Async DMA begin | v2, v3 |
| `async-copy-done` | Async DMA complete | v2, v3 |
| `[2,...]` shape | Double buffer | v2, v3 |
| `constant(0)` / `constant(1)` | Buffer index | v2, v3 |
| `f8e4m3fn.*dot` | Native fp8 matmul | v1, v2, v3 |

Remember: HLO verification is about **correctness**, not performance. The real test is benchmarking on actual TPU hardware!
