# SPDX-License-Identifier: Apache-2.0
"""Tests for 2D fp8 quantized matmul kernel."""

import jax
import jax.numpy as jnp
from absl.testing import absltest, parameterized
from jax._src import test_util as jtu

from tpu_inference.kernels.fp8_quantized_matmul_2d import (
    fp8_quantized_matmul_2d_kernel,
    quantize_tensor_2d,
    xla_quantized_matmul_2d,
    get_tuned_block_sizes,
)

jax.config.parse_flags_with_absl()


@jtu.with_config(jax_numpy_dtype_promotion="standard")
class Fp8QuantizedMatmul2DKernelTest(jtu.JaxTestCase):

    def setUp(self):
        super().setUp()
        # This kernel is designed for Ironwood TPU with fp8 support
        # For testing, we'll use TPUv6+ which has similar capabilities
        if not jtu.is_device_tpu_at_least(6):
            self.skipTest("Expect TPUv6+ for fp8 support")

    def _test_fp8_quantized_matmul_2d(
        self,
        dtype: jnp.dtype,
        q_dtype: jnp.dtype,
        bs: int,
        n_input_features: int,
        n_output_features: int,
        quant_block_size: int,
        quantize_activation: bool,
        tuned_value=None,
        atol=1.0,
        rtol=1.0,
    ):
        """Test helper for 2D quantized matmul.

        Args:
            dtype: Input data type
            q_dtype: Quantization data type
            bs: Batch size
            n_input_features: Input feature dimension
            n_output_features: Output feature dimension
            quant_block_size: Size of 2D quantization blocks (128, 256, or 512)
            quantize_activation: Whether to quantize activations
            tuned_value: Optional tuned block size configuration
            atol: Absolute tolerance for comparison
            rtol: Relative tolerance for comparison
        """

        # Ensure dimensions are divisible by quant_block_size
        bs = ((bs + quant_block_size - 1) // quant_block_size) * quant_block_size
        n_input_features = ((n_input_features + quant_block_size - 1) // quant_block_size) * quant_block_size
        n_output_features = ((n_output_features + quant_block_size - 1) // quant_block_size) * quant_block_size

        prng_key = jax.random.key(1234)
        k0, k1 = jax.random.split(prng_key, 2)

        # Generate random inputs
        x = jax.random.uniform(
            k0,
            (bs, n_input_features),
            dtype=dtype,
            minval=-1,
            maxval=1,
        )
        w = jax.random.uniform(
            k1,
            (n_output_features, n_input_features),
            dtype=dtype,
            minval=-1,
            maxval=1,
        )

        # Quantize weights with 2D blocks
        w_q, w_scale = quantize_tensor_2d(
            w,
            q_dtype,
            block_size_m=quant_block_size,
            block_size_n=quant_block_size,
        )

        # Verify scale shape
        expected_scale_shape = (
            n_output_features // quant_block_size,
            n_input_features // quant_block_size,
        )
        self.assertEqual(w_scale.shape, expected_scale_shape)

        # Run kernel
        x_q_dtype = w_q.dtype if quantize_activation else dtype
        output = fp8_quantized_matmul_2d_kernel(
            x,
            w_q,
            w_scale,
            x_q_dtype=x_q_dtype,
            quant_block_size=quant_block_size,
            tuned_value=tuned_value,
        )

        # Compute reference output
        expected = xla_quantized_matmul_2d(
            x,
            w_q,
            w_scale,
            x_quantize=quantize_activation,
            quant_block_size=quant_block_size,
        )

        # Compare results
        self.assertAllClose(
            output,
            expected,
            rtol=rtol,
            atol=atol,
            check_dtypes=True,
        )

    @parameterized.product(
        dtype=[jnp.bfloat16],
        q_dtype=[jnp.float8_e4m3fn],
        bs=[128, 256, 512],
        n_input_features=[128, 256, 512],
        n_output_features=[128, 256, 512],
        quant_block_size=[128, 256],
        quantize_activation=[True, False],
    )
    def test_fp8_quantized_matmul_2d_various_shapes(
        self,
        dtype: jnp.dtype,
        q_dtype: jnp.dtype,
        bs: int,
        n_input_features: int,
        n_output_features: int,
        quant_block_size: int,
        quantize_activation: bool,
    ):
        """Test 2D quantized matmul with various input shapes."""
        self._test_fp8_quantized_matmul_2d(
            dtype,
            q_dtype,
            bs,
            n_input_features,
            n_output_features,
            quant_block_size,
            quantize_activation=quantize_activation,
            tuned_value=None,
        )

    @parameterized.product(
        dtype=[jnp.bfloat16],
        q_dtype=[jnp.float8_e4m3fn],
        quant_block_size=[128, 256, 512],
        quantize_activation=[True],
    )
    def test_fp8_quantized_matmul_2d_common_sizes(
        self,
        dtype: jnp.dtype,
        q_dtype: jnp.dtype,
        quant_block_size: int,
        quantize_activation: bool,
    ):
        """Test common matrix sizes with different quantization block sizes."""
        # Common sizes for LLM workloads
        test_configs = [
            (1024, 1024, 1024),
            (1024, 2048, 1024),
            (2048, 2048, 2048),
            (1024, 4096, 2048),
        ]

        for bs, n_out, n_in in test_configs:
            with self.subTest(bs=bs, n_out=n_out, n_in=n_in):
                self._test_fp8_quantized_matmul_2d(
                    dtype,
                    q_dtype,
                    bs,
                    n_in,
                    n_out,
                    quant_block_size,
                    quantize_activation=quantize_activation,
                    tuned_value=None,
                )

    @parameterized.parameters(
        (jnp.bfloat16, jnp.float8_e4m3fn, 1024, 1024, 1024, 128, True),
        (jnp.bfloat16, jnp.float8_e4m3fn, 2048, 2048, 2048, 256, True),
        (jnp.bfloat16, jnp.float8_e4m3fn, 4096, 4096, 4096, 512, True),
    )
    def test_fp8_quantized_matmul_2d_use_tuned_block_sizes(
        self,
        dtype: jnp.dtype,
        q_dtype: jnp.dtype,
        bs: int,
        n_input_features: int,
        n_output_features: int,
        quant_block_size: int,
        quantize_activation: bool,
    ):
        """Test using tuned block sizes from the database."""
        # Get tuned block sizes
        tuned_value = get_tuned_block_sizes(
            n_batch=bs,
            n_out=n_output_features,
            n_in=n_input_features,
            x_q_dtype=jnp.dtype(q_dtype).name if quantize_activation else jnp.dtype(dtype).name,
            w_q_dtype=jnp.dtype(q_dtype).name,
            quant_block_size=quant_block_size,
        )

        self._test_fp8_quantized_matmul_2d(
            dtype,
            q_dtype,
            bs,
            n_input_features,
            n_output_features,
            quant_block_size,
            quantize_activation=quantize_activation,
            tuned_value=tuned_value,
        )

    def test_quantize_tensor_2d(self):
        """Test 2D quantization function."""
        x = jnp.ones((512, 512), dtype=jnp.bfloat16)
        quant_block_size = 128

        x_q, x_scale = quantize_tensor_2d(
            x,
            jnp.float8_e4m3fn,
            block_size_m=quant_block_size,
            block_size_n=quant_block_size,
        )

        # Verify shapes
        self.assertEqual(x_q.shape, x.shape)
        expected_scale_shape = (
            512 // quant_block_size,
            512 // quant_block_size,
        )
        self.assertEqual(x_scale.shape, expected_scale_shape)

        # Verify dtype
        self.assertEqual(x_q.dtype, jnp.float8_e4m3fn)
        self.assertEqual(x_scale.dtype, jnp.float32)


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
