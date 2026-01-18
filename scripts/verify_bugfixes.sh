#!/bin/bash
# Verification script for v2/v3 bug fixes
# This verifies that the memory space and semaphore fixes are working

set -e

echo "=========================================="
echo "V2/V3 Bug Fix Verification"
echo "=========================================="
echo ""

# Check if compilation output exists
if [ ! -d "compilation_output" ]; then
    echo "No compilation output found. Generating..."
    python scripts/inspect_fp8_2d_kernel_compilation.py
fi

echo "Checking v2 (Manual Async DMA) fixes..."
echo "=========================================="
echo ""

# V2 Fix Verification
echo "1. Checking HBM memory space for inputs..."
if grep -q "memory_space=0" compilation_output/v2_manual_async_dma_hlo.txt; then
    echo "   ✅ Found memory_space=0 (HBM) - inputs stay in HBM"
else
    echo "   ❌ No memory_space=0 found - inputs may be auto-fetched"
fi

echo ""
echo "2. Checking for async DMA operations..."
ASYNC_START=$(grep -c "async-copy-start\|async-start" compilation_output/v2_manual_async_dma_hlo.txt || echo "0")
ASYNC_DONE=$(grep -c "async-copy-done\|async-done" compilation_output/v2_manual_async_dma_hlo.txt || echo "0")

if [ "$ASYNC_START" -gt "0" ] && [ "$ASYNC_DONE" -gt "0" ]; then
    echo "   ✅ Found $ASYNC_START async starts and $ASYNC_DONE completions"
    if [ "$ASYNC_START" -eq "$ASYNC_DONE" ]; then
        echo "   ✅ Balanced async operations"
    else
        echo "   ⚠️  Imbalanced: $ASYNC_START starts vs $ASYNC_DONE completions"
    fi
else
    echo "   ❌ No async operations found - manual DMA not working"
fi

echo ""
echo "3. Checking for double buffering (leading dim = 2)..."
DOUBLE_BUF=$(grep -c "\[2," compilation_output/v2_manual_async_dma_hlo.txt || echo "0")
if [ "$DOUBLE_BUF" -gt "0" ]; then
    echo "   ✅ Found $DOUBLE_BUF double-buffered arrays"
    echo "   Sample:"
    grep "\[2," compilation_output/v2_manual_async_dma_hlo.txt | head -3 | sed 's/^/      /'
else
    echo "   ❌ No double-buffered arrays found"
fi

echo ""
echo "4. Checking for DMA semaphores..."
if grep -q "semaphore\|Semaphore" compilation_output/v2_manual_async_dma_hlo.txt; then
    echo "   ✅ Semaphores found in HLO"
else
    echo "   ⚠️  No explicit semaphore references (may be implicit)"
fi

echo ""
echo ""
echo "Checking v3 (SMEM Scales) fixes..."
echo "=========================================="
echo ""

# V3 Fix Verification
echo "1. Checking HBM memory space for inputs (same as v2)..."
if grep -q "memory_space=0" compilation_output/v3_smem_scales_hlo.txt; then
    echo "   ✅ Found memory_space=0 (HBM)"
else
    echo "   ❌ No memory_space=0 found"
fi

echo ""
echo "2. Checking for SMEM usage (memory_space=2)..."
SMEM_COUNT=$(grep -c "memory_space=2\|S(2)" compilation_output/v3_smem_scales_hlo.txt || echo "0")
if [ "$SMEM_COUNT" -gt "0" ]; then
    echo "   ✅ Found $SMEM_COUNT SMEM references - v3 optimization working!"
    echo "   Looking for scale patterns in SMEM:"
    grep -i "scale.*memory_space=2\|scale.*S(2)" compilation_output/v3_smem_scales_hlo.txt | head -3 | sed 's/^/      /' || echo "      (scales may not have explicit markers)"
else
    echo "   ❌ No SMEM usage found - v3 optimization NOT working"
fi

echo ""
echo "3. Checking for VMEM usage (memory_space=1) for data..."
VMEM_COUNT=$(grep -c "memory_space=1\|S(1)" compilation_output/v3_smem_scales_hlo.txt || echo "0")
if [ "$VMEM_COUNT" -gt "0" ]; then
    echo "   ✅ Found $VMEM_COUNT VMEM references - data blocks in VMEM"
else
    echo "   ⚠️  No explicit VMEM references"
fi

echo ""
echo "4. Checking for async DMA (same as v2)..."
ASYNC_START_V3=$(grep -c "async-copy-start\|async-start" compilation_output/v3_smem_scales_hlo.txt || echo "0")
ASYNC_DONE_V3=$(grep -c "async-copy-done\|async-done" compilation_output/v3_smem_scales_hlo.txt || echo "0")

if [ "$ASYNC_START_V3" -gt "0" ] && [ "$ASYNC_DONE_V3" -gt "0" ]; then
    echo "   ✅ Found $ASYNC_START_V3 async starts and $ASYNC_DONE_V3 completions"
else
    echo "   ❌ No async operations found"
fi

echo ""
echo ""
echo "=========================================="
echo "Summary"
echo "=========================================="
echo ""

# V2 Summary
echo "V2 (Manual Async DMA):"
V2_SCORE=0
[ "$ASYNC_START" -gt "0" ] && V2_SCORE=$((V2_SCORE + 1))
[ "$ASYNC_DONE" -gt "0" ] && V2_SCORE=$((V2_SCORE + 1))
[ "$DOUBLE_BUF" -gt "0" ] && V2_SCORE=$((V2_SCORE + 1))
grep -q "memory_space=0" compilation_output/v2_manual_async_dma_hlo.txt && V2_SCORE=$((V2_SCORE + 1))

if [ "$V2_SCORE" -ge "3" ]; then
    echo "  ✅ PASS ($V2_SCORE/4 checks) - Bug fixes appear to be working"
else
    echo "  ❌ FAIL ($V2_SCORE/4 checks) - Some fixes may not be working"
fi

# V3 Summary
echo ""
echo "V3 (SMEM Scales):"
V3_SCORE=0
[ "$SMEM_COUNT" -gt "0" ] && V3_SCORE=$((V3_SCORE + 1))
[ "$VMEM_COUNT" -gt "0" ] && V3_SCORE=$((V3_SCORE + 1))
[ "$ASYNC_START_V3" -gt "0" ] && V3_SCORE=$((V3_SCORE + 1))
grep -q "memory_space=0" compilation_output/v3_smem_scales_hlo.txt && V3_SCORE=$((V3_SCORE + 1))

if [ "$V3_SCORE" -ge "3" ]; then
    echo "  ✅ PASS ($V3_SCORE/4 checks) - Bug fixes appear to be working"
else
    echo "  ❌ FAIL ($V3_SCORE/4 checks) - Some fixes may not be working"
fi

echo ""
echo "=========================================="
echo ""

if [ "$V2_SCORE" -ge "3" ] && [ "$V3_SCORE" -ge "3" ]; then
    echo "✅ Both v2 and v3 look good! Fixes verified."
    echo ""
    echo "Next steps:"
    echo "  1. Run on actual TPU to get full LLO"
    echo "  2. Benchmark all three versions"
    echo "  3. Compare performance"
    exit 0
else
    echo "⚠️  Some issues detected. Review HLO output manually."
    echo ""
    echo "Manual inspection:"
    echo "  python scripts/analyze_hlo.py compilation_output/*_hlo.txt"
    exit 1
fi
