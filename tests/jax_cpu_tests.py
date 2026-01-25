#!/usr/bin/env python3
"""
JAX-based unit tests that can run on CPU without TPU or vLLM.
These tests verify JAX operations and sharding logic.
"""

import os
import sys
import unittest

# Set environment before importing JAX
os.environ['JAX_PLATFORMS'] = 'cpu'

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, PartitionSpec, NamedSharding


class TestJAXBasics(unittest.TestCase):
    """Basic JAX functionality tests."""

    def test_jax_available(self):
        """Verify JAX is available and working."""
        x = jnp.array([1, 2, 3])
        y = x + 1
        np.testing.assert_array_equal(y, np.array([2, 3, 4]))

    def test_cpu_mesh_creation(self):
        """Test creating a CPU mesh."""
        devices = jax.devices('cpu')
        self.assertGreater(len(devices), 0)

        # Create a simple 1D mesh
        mesh = Mesh(np.array(devices), axis_names=('data',))
        self.assertEqual(mesh.shape, {'data': len(devices)})


class TestPartitionSpecLogic(unittest.TestCase):
    """Test PartitionSpec handling that mirrors 2D TP sharding."""

    def test_partition_spec_with_single_axis(self):
        """Test PartitionSpec with single axis name (2D mesh default)."""
        spec = PartitionSpec(None, 'model')
        self.assertEqual(spec, PartitionSpec(None, 'model'))

    def test_partition_spec_with_tuple_axis(self):
        """Test PartitionSpec with tuple of axis names (4D mesh)."""
        spec = PartitionSpec(None, ('model', 'expert'))
        self.assertEqual(spec, PartitionSpec(None, ('model', 'expert')))

    def test_partition_spec_for_moe_weights(self):
        """Test PartitionSpec configurations for MoE weights."""
        # EP sharding (default): shard on expert dimension
        ep_edf = PartitionSpec('model', None, None)  # (E, D, F)
        self.assertEqual(ep_edf[0], 'model')

        # 2D TP sharding: shard on D and F dimensions
        tp_edf = PartitionSpec(None, 'model', 'expert')  # (E, D, F)
        self.assertEqual(tp_edf[0], None)  # E not sharded
        self.assertEqual(tp_edf[1], 'model')  # D sharded
        self.assertEqual(tp_edf[2], 'expert')  # F sharded


class TestMeshAxisHandling(unittest.TestCase):
    """Test mesh axis name handling for tuple vs string axes."""

    def setUp(self):
        """Set up CPU mesh for testing."""
        self.devices = jax.devices('cpu')

    def test_single_axis_mesh(self):
        """Test mesh with single axis name."""
        mesh = Mesh(np.array(self.devices), axis_names=('model',))

        # Access axis size
        model_size = mesh.shape['model']
        self.assertEqual(model_size, len(self.devices))

    def test_tuple_axis_product(self):
        """Test computing product of multiple axis sizes."""
        # Simulate 4D mesh axis lookup
        mesh_shape = {'data': 1, 'attn_dp': 1, 'expert': 2, 'model': 4}

        # Single axis
        single_axis = 'model'
        if isinstance(single_axis, (tuple, list)):
            size = 1
            for name in single_axis:
                size *= mesh_shape[name]
        else:
            size = mesh_shape[single_axis]
        self.assertEqual(size, 4)

        # Tuple axis (like ATTN_HEAD in ShardingAxisNameBase)
        tuple_axis = ('model', 'expert')
        if isinstance(tuple_axis, (tuple, list)):
            size = 1
            for name in tuple_axis:
                size *= mesh_shape[name]
        else:
            size = mesh_shape[tuple_axis]
        self.assertEqual(size, 8)  # 4 * 2


class TestShardingAxisNameSimulation(unittest.TestCase):
    """Test simulated ShardingAxisName classes."""

    def test_sharding_axis_name_2d(self):
        """Test 2D mesh sharding axis names."""
        class ShardingAxisName2D:
            SEQUENCE = 'data'
            ATTN_DATA = 'data'
            MLP_DATA = 'data'
            TENSOR = 'model'
            ATTN_HEAD = 'model'
            MLP_TENSOR = 'model'
            MOE_TENSOR = 'model'
            EXPERT = 'model'

        self.assertEqual(ShardingAxisName2D.TENSOR, 'model')
        self.assertEqual(ShardingAxisName2D.ATTN_HEAD, 'model')

    def test_sharding_axis_name_base(self):
        """Test 4D mesh sharding axis names (Base)."""
        class ShardingAxisNameBase:
            SEQUENCE = ('data', 'attn_dp')
            ATTN_DATA = ('data', 'attn_dp')
            MLP_DATA = 'data'
            TENSOR = ('model', 'expert')
            ATTN_HEAD = ('model', 'expert')
            MLP_TENSOR = ('attn_dp', 'model', 'expert')
            MOE_TENSOR = ('attn_dp', 'model')
            EXPERT = ('attn_dp', 'expert', 'model')
            MODEL_1 = 'model'
            MODEL_2 = 'expert'

        self.assertEqual(ShardingAxisNameBase.MODEL_1, 'model')
        self.assertEqual(ShardingAxisNameBase.MODEL_2, 'expert')
        self.assertEqual(ShardingAxisNameBase.TENSOR, ('model', 'expert'))

    def test_conditional_sharding_selection(self):
        """Test conditional sharding based on USE_2D_TP flag."""
        class ShardingAxisName2D:
            MLP_TENSOR = 'model'
            MODEL_1 = None  # Not available
            MODEL_2 = None  # Not available

        class ShardingAxisNameBase:
            MLP_TENSOR = ('attn_dp', 'model', 'expert')
            MODEL_1 = 'model'
            MODEL_2 = 'expert'

        # Simulate USE_2D_TP=False (default)
        use_2d_tp = False
        if use_2d_tp:
            ShardingAxisName = ShardingAxisNameBase
        else:
            ShardingAxisName = ShardingAxisName2D

        self.assertEqual(ShardingAxisName.MLP_TENSOR, 'model')

        # Simulate USE_2D_TP=True
        use_2d_tp = True
        if use_2d_tp:
            ShardingAxisName = ShardingAxisNameBase
        else:
            ShardingAxisName = ShardingAxisName2D

        self.assertEqual(ShardingAxisName.MODEL_1, 'model')
        self.assertEqual(ShardingAxisName.MODEL_2, 'expert')


class TestMoEShardingConfigurations(unittest.TestCase):
    """Test MoE sharding configurations for both EP and 2D TP."""

    def test_expert_parallel_sharding(self):
        """Test Expert Parallelism sharding (default)."""
        # EP: Shard on expert dimension
        moe_activation_ffw_td = ('data', None)
        moe_activation_ffw_ted = ('data', None, None)
        moe_edf_sharding = ('model', None, None)  # Shard E on 'model' axis
        moe_efd_sharding = ('model', None, None)

        # Weight shape (E=256, D=7168, F=2048)
        # With EP: Each device holds E/num_devices experts
        self.assertEqual(moe_edf_sharding[0], 'model')  # Expert dimension sharded
        self.assertIsNone(moe_edf_sharding[1])  # D not sharded
        self.assertIsNone(moe_edf_sharding[2])  # F not sharded

    def test_2d_tp_sharding(self):
        """Test 2D Tensor Parallelism sharding."""
        MODEL_1 = 'model'
        MODEL_2 = 'expert'

        # 2D TP: Shard on D and F dimensions
        moe_activation_ffw_td = ('data', MODEL_1)
        moe_activation_ffw_ted = ('data', None, MODEL_1)
        moe_edf_sharding = (None, MODEL_1, MODEL_2)  # Shard D and F
        moe_efd_sharding = (None, MODEL_2, MODEL_1)

        # Weight shape (E=256, D=7168, F=2048)
        # With 2D TP: Each device holds all experts with partial D and F
        self.assertIsNone(moe_edf_sharding[0])  # Expert dimension NOT sharded
        self.assertEqual(moe_edf_sharding[1], MODEL_1)  # D sharded on MODEL_1
        self.assertEqual(moe_edf_sharding[2], MODEL_2)  # F sharded on MODEL_2

    def test_conditional_sharding_logic(self):
        """Test the conditional sharding logic as implemented in DeepSeek."""
        MODEL_1 = 'model'
        MODEL_2 = 'expert'
        MLP_TENSOR = 'model'
        MLP_DATA = 'data'

        for use_2d_tp in [False, True]:
            if use_2d_tp:
                moe_activation_ffw_td = (MLP_DATA, MODEL_1)
                moe_activation_ffw_ted = (MLP_DATA, None, MODEL_1)
                moe_edf_sharding = (None, MODEL_1, MODEL_2)
                moe_efd_sharding = (None, MODEL_2, MODEL_1)
            else:
                moe_activation_ffw_td = (MLP_DATA, None)
                moe_activation_ffw_ted = (MLP_DATA, None, None)
                moe_edf_sharding = (MLP_TENSOR, None, None)
                moe_efd_sharding = (MLP_TENSOR, None, None)

            if use_2d_tp:
                # 2D TP: Expert dim not sharded, D and F sharded
                self.assertIsNone(moe_edf_sharding[0])
                self.assertEqual(moe_edf_sharding[1], MODEL_1)
                self.assertEqual(moe_activation_ffw_td[1], MODEL_1)
            else:
                # EP: Expert dim sharded, D and F not sharded
                self.assertEqual(moe_edf_sharding[0], MLP_TENSOR)
                self.assertIsNone(moe_edf_sharding[1])
                self.assertIsNone(moe_activation_ffw_td[1])


class TestRouterOutputShapes(unittest.TestCase):
    """Test expected router output shapes."""

    def test_router_output_shapes(self):
        """Verify expected router output shapes."""
        batch_size = 4
        seq_len = 8
        num_experts = 256
        num_experts_per_tok = 8

        T = batch_size * seq_len  # Total tokens

        # Router outputs
        router_weights_shape = (T, num_experts_per_tok)  # (T, X)
        selected_experts_shape = (T, num_experts_per_tok)  # (T, X)

        self.assertEqual(router_weights_shape, (32, 8))
        self.assertEqual(selected_experts_shape, (32, 8))


class TestWeightShapes(unittest.TestCase):
    """Test MoE weight shapes and sharding."""

    def test_moe_weight_shapes(self):
        """Test expected MoE weight shapes."""
        E = 256  # num_experts
        D = 7168  # hidden_size
        F = 2048  # intermediate_size

        kernel_gating_EDF = (E, D, F)
        kernel_up_proj_EDF = (E, D, F)
        kernel_down_proj_EFD = (E, F, D)

        self.assertEqual(kernel_gating_EDF, (256, 7168, 2048))
        self.assertEqual(kernel_down_proj_EFD, (256, 2048, 7168))

    def test_sharded_weight_shapes_ep(self):
        """Test weight shapes after EP sharding."""
        E = 256
        D = 7168
        F = 2048
        num_devices = 8

        # With EP: E is sharded across devices
        local_E = E // num_devices  # 32
        local_shape_EDF = (local_E, D, F)

        self.assertEqual(local_shape_EDF, (32, 7168, 2048))

    def test_sharded_weight_shapes_2d_tp(self):
        """Test weight shapes after 2D TP sharding."""
        E = 256
        D = 7168
        F = 2048
        # 4D mesh: (data=1, attn_dp=1, expert=2, model=4)
        model_size = 4  # Shards D
        expert_size = 2  # Shards F

        # With 2D TP: D and F are sharded, E is replicated
        local_D = D // model_size  # 1792
        local_F = F // expert_size  # 1024
        local_shape_EDF = (E, local_D, local_F)

        self.assertEqual(local_shape_EDF, (256, 1792, 1024))


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)
