# SPDX-License-Identifier: Apache-2.0
"""2D fp8 quantized matmul kernel for Ironwood TPU.

This module provides an efficient Pallas kernel for performing matrix multiplication
with 2D block-wise fp8 quantization on TPU. The kernel is optimized for Ironwood TPU
which features:
- 64MB VMEM
- 256x256 MXU with native fp8 math support

Key features:
- 2D block-wise quantization for both weights and activations
- Supports quantization blocks of size 128x128, 256x256, and 512x512
- Optimized memory layout ensuring divisibility by (8x128) for TPU constraints
- fp8_e4m3fn quantization with f32 scales

Example usage:
    ```python
    import jax.numpy as jnp
    from tpu_inference.kernels.fp8_quantized_matmul_2d import (
        fp8_quantized_matmul_2d_kernel,
        quantize_tensor_2d,
    )

    # Prepare inputs
    x = jnp.ones((1024, 2048), dtype=jnp.bfloat16)
    w = jnp.ones((4096, 2048), dtype=jnp.bfloat16)

    # Quantize weights with 2D blocks
    w_q, w_scale = quantize_tensor_2d(
        w, jnp.float8_e4m3fn, block_size_m=256, block_size_n=256
    )

    # Run kernel (activations will be quantized automatically)
    out = fp8_quantized_matmul_2d_kernel(
        x,
        w_q,
        w_scale,
        x_q_dtype=jnp.float8_e4m3fn,
        quant_block_size=256,
    )
    ```
"""

from tpu_inference.kernels.fp8_quantized_matmul_2d.kernel import (
    fp8_quantized_matmul_2d_kernel,
)
from tpu_inference.kernels.fp8_quantized_matmul_2d.tuned_block_sizes import (
    TunedValue,
    get_device_vmem_limit,
    get_tuned_block_sizes,
)
from tpu_inference.kernels.fp8_quantized_matmul_2d.util import (
    quantize_tensor_2d,
    xla_quantized_matmul_2d,
)

__all__ = [
    "fp8_quantized_matmul_2d_kernel",
    "quantize_tensor_2d",
    "xla_quantized_matmul_2d",
    "TunedValue",
    "get_tuned_block_sizes",
    "get_device_vmem_limit",
]
