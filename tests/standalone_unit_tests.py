#!/usr/bin/env python3
"""
Standalone unit tests that can run without TPU or vLLM dependencies.
These tests verify the core logic of the 2D TP sharding implementation.
"""

import os
import sys
import unittest

# Set environment to avoid TPU-specific initialization issues
os.environ.setdefault('JAX_PLATFORMS', 'cpu')


class TestShardingAxisNames(unittest.TestCase):
    """Test that sharding axis name classes have correct attributes."""

    def test_sharding_axis_name_base_attributes(self):
        """Verify ShardingAxisNameBase has all required attributes."""
        # Import directly from the module file to avoid full package init
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "sharding_module",
            "/home/user/tpu-inference/tpu_inference/layers/common/sharding.py"
        )
        # Can't import directly due to dependencies, so we'll parse
        with open("/home/user/tpu-inference/tpu_inference/layers/common/sharding.py", 'r') as f:
            content = f.read()

        # Verify ShardingAxisNameBase has MODEL_1 and MODEL_2
        self.assertIn("MODEL_1 = 'model'", content)
        self.assertIn("MODEL_2 = 'expert'", content)

        # Verify TENSOR attribute exists in Base
        self.assertIn("TENSOR = ('model', 'expert')", content)

        # Verify ATTN_HEAD is a tuple in Base
        self.assertIn("ATTN_HEAD = ('model', 'expert')", content)

    def test_sharding_axis_name_2d_attributes(self):
        """Verify ShardingAxisName2D has all required attributes."""
        with open("/home/user/tpu-inference/tpu_inference/layers/common/sharding.py", 'r') as f:
            content = f.read()

        # Find ShardingAxisName2D class section
        class_start = content.find("class ShardingAxisName2D:")
        class_end = content.find("\n\ntry:", class_start)
        class_content = content[class_start:class_end]

        # Verify TENSOR attribute exists in 2D class
        self.assertIn("TENSOR = 'model'", class_content,
                      "ShardingAxisName2D must have TENSOR attribute for compilation_manager.py")

        # Verify other essential attributes
        self.assertIn("ATTN_HEAD = 'model'", class_content)
        self.assertIn("MLP_TENSOR = 'model'", class_content)
        self.assertIn("MLP_DATA = 'data'", class_content)

    def test_use_2d_tp_selection_logic(self):
        """Verify the USE_2D_TP flag selection logic is present."""
        with open("/home/user/tpu-inference/tpu_inference/layers/common/sharding.py", 'r') as f:
            content = f.read()

        # Verify selection logic exists
        self.assertIn("_use_2d_tp_sharding = envs.USE_2D_TP", content)
        self.assertIn("if _use_2d_tp_sharding or _use_base_sharding:", content)
        self.assertIn("ShardingAxisName = ShardingAxisNameBase", content)


class TestEnvsModule(unittest.TestCase):
    """Test that envs module has the USE_2D_TP variable."""

    def test_use_2d_tp_defined(self):
        """Verify USE_2D_TP is defined in envs.py."""
        with open("/home/user/tpu-inference/tpu_inference/envs.py", 'r') as f:
            content = f.read()

        self.assertIn('"USE_2D_TP"', content)
        self.assertIn('env_bool("USE_2D_TP"', content)


class TestDeepSeekV3Model(unittest.TestCase):
    """Test DeepSeek V3 model has correct conditional sharding."""

    def test_conditional_2d_tp_sharding(self):
        """Verify DeepSeek model uses conditional sharding based on USE_2D_TP."""
        with open("/home/user/tpu-inference/tpu_inference/models/jax/deepseek_v3.py", 'r') as f:
            content = f.read()

        # Verify envs is imported
        self.assertIn("from tpu_inference import envs", content)

        # Verify conditional sharding logic exists
        self.assertIn("if envs.USE_2D_TP:", content)

        # Verify 2D TP sharding uses MODEL_1 and MODEL_2
        self.assertIn("ShardingAxisName.MODEL_1", content)
        self.assertIn("ShardingAxisName.MODEL_2", content)

        # Verify fallback to standard sharding
        self.assertIn("moe_edf_sharding = (ShardingAxisName.MLP_TENSOR, None, None)", content)
        self.assertIn("moe_efd_sharding = (ShardingAxisName.MLP_TENSOR, None, None)", content)

    def test_moe_sharding_variables_used(self):
        """Verify MoE uses the conditional sharding variables."""
        with open("/home/user/tpu-inference/tpu_inference/models/jax/deepseek_v3.py", 'r') as f:
            content = f.read()

        # Verify the variables are actually used in MoE construction
        self.assertIn("activation_ffw_td=moe_activation_ffw_td", content)
        self.assertIn("activation_ffw_ted=moe_activation_ffw_ted", content)
        self.assertIn("edf_sharding=moe_edf_sharding", content)
        self.assertIn("efd_sharding=moe_efd_sharding", content)


class TestMoEFunctionalAPI(unittest.TestCase):
    """Test MoE implementation uses functional API from PR 1287."""

    def test_moe_imports_functional_api(self):
        """Verify moe.py imports functional forward functions."""
        with open("/home/user/tpu-inference/tpu_inference/layers/jax/moe/moe.py", 'r') as f:
            content = f.read()

        # Verify functional API imports
        self.assertIn("from tpu_inference.layers.jax.moe.dense_moe import", content)
        self.assertIn("dense_moe_fwd", content)
        self.assertIn("dense_moe_fwd_preapply_router_weights", content)
        self.assertIn("from tpu_inference.layers.jax.moe.sparse_moe import sparse_moe_distributed_fwd", content)

    def test_dense_moe_function_signature(self):
        """Verify dense_moe.py has functional API with moe_instance parameter."""
        with open("/home/user/tpu-inference/tpu_inference/layers/jax/moe/dense_moe.py", 'r') as f:
            content = f.read()

        # Verify function signatures take moe_instance as first parameter
        self.assertIn("def dense_moe_fwd(moe_instance,", content)
        self.assertIn("def dense_moe_fwd_preapply_router_weights(moe_instance,", content)

    def test_sparse_moe_function_signature(self):
        """Verify sparse_moe.py has functional API with moe_instance parameter."""
        with open("/home/user/tpu-inference/tpu_inference/layers/jax/moe/sparse_moe.py", 'r') as f:
            content = f.read()

        # Verify function signature takes moe_instance as first parameter
        self.assertIn("def sparse_moe_distributed_fwd(\n    moe_instance,", content)

    def test_moe_uses_functional_api(self):
        """Verify MoE.__call__ uses the functional API correctly."""
        with open("/home/user/tpu-inference/tpu_inference/layers/jax/moe/moe.py", 'r') as f:
            content = f.read()

        # Verify functional calls with self as first arg
        self.assertIn("dense_moe_fwd_preapply_router_weights(\n                            self,", content)
        self.assertIn("dense_moe_fwd(self,", content)


class TestKVCacheHandlesTupleAxis(unittest.TestCase):
    """Test KV cache code handles tuple axis names correctly."""

    def test_kv_cache_handles_tuple_axis(self):
        """Verify kv_cache.py handles tuple axis names."""
        with open("/home/user/tpu-inference/tpu_inference/runner/kv_cache.py", 'r') as f:
            content = f.read()

        # Verify tuple/list handling
        self.assertIn("isinstance(axis_name, (tuple, list))", content)
        self.assertIn("math.prod(mesh.shape[name] for name in axis_name)", content)

    def test_kv_cache_manager_handles_tuple_axis(self):
        """Verify kv_cache_manager.py handles tuple axis names."""
        with open("/home/user/tpu-inference/tpu_inference/runner/kv_cache_manager.py", 'r') as f:
            content = f.read()

        # Verify tuple/list handling
        self.assertIn("isinstance(tp_axis_name, (tuple, list))", content)
        self.assertIn("math.prod(self.runner.mesh.shape[name]", content)


class TestCompilationManagerUsesTensor(unittest.TestCase):
    """Test compilation_manager.py uses ShardingAxisName.TENSOR."""

    def test_uses_tensor_attribute(self):
        """Verify compilation_manager.py uses ShardingAxisName.TENSOR."""
        with open("/home/user/tpu-inference/tpu_inference/runner/compilation_manager.py", 'r') as f:
            content = f.read()

        # Verify TENSOR is used (not hardcoded 'model')
        self.assertIn("ShardingAxisName.TENSOR", content)
        self.assertIn("from tpu_inference.layers.common.sharding import ShardingAxisName", content)


class TestAttentionUsesShardingAxisName(unittest.TestCase):
    """Test attention code uses ShardingAxisName instead of hardcoded strings."""

    def test_deepseek_attention_uses_sharding_axis(self):
        """Verify deepseek_v3_attention.py uses ShardingAxisName.ATTN_HEAD."""
        with open("/home/user/tpu-inference/tpu_inference/layers/jax/attention/deepseek_v3_attention.py", 'r') as f:
            content = f.read()

        # Verify ATTN_HEAD is used for KV cache sharding
        self.assertIn("ShardingAxisName.ATTN_HEAD", content)


class TestDocumentation(unittest.TestCase):
    """Test documentation files exist and have key content."""

    def test_moe_2d_tp_doc_exists(self):
        """Verify MoE 2D TP documentation exists."""
        doc_path = "/home/user/tpu-inference/docs/deepseek_moe_2d_tp.md"
        self.assertTrue(os.path.exists(doc_path), "MoE 2D TP documentation should exist")

        with open(doc_path, 'r') as f:
            content = f.read()

        # Check key sections
        self.assertIn("2D Tensor Parallelism", content)
        self.assertIn("Expert Parallelism", content)
        self.assertIn("USE_2D_TP", content)

    def test_architecture_doc_exists(self):
        """Verify architecture documentation exists."""
        doc_path = "/home/user/tpu-inference/docs/technical_architecture.md"
        self.assertTrue(os.path.exists(doc_path), "Architecture documentation should exist")

        with open(doc_path, 'r') as f:
            content = f.read()

        # Check key sections
        self.assertIn("MoE Implementation", content)
        self.assertIn("Sharding", content)


class TestMoEBackendSelection(unittest.TestCase):
    """Test MoE backend selection logic."""

    def test_backend_enum_exists(self):
        """Verify MoEBackend enum is defined."""
        with open("/home/user/tpu-inference/tpu_inference/layers/jax/moe/utils.py", 'r') as f:
            content = f.read()

        self.assertIn("class MoEBackend(enum.Enum):", content)
        self.assertIn("FUSED_MOE", content)
        self.assertIn("VLLM_MOE", content)
        self.assertIn("DENSE_MAT", content)
        self.assertIn("MEGABLX_GMM", content)
        self.assertIn("RAGGED_DOT", content)

    def test_select_moe_backend_function(self):
        """Verify select_moe_backend function exists."""
        with open("/home/user/tpu-inference/tpu_inference/layers/jax/moe/utils.py", 'r') as f:
            content = f.read()

        self.assertIn("def select_moe_backend():", content)
        # Verify it checks environment variables
        self.assertIn("envs.USE_MOE_EP_KERNEL", content)
        self.assertIn("envs.USE_VLLM_MOE_KERNEL", content)


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)
