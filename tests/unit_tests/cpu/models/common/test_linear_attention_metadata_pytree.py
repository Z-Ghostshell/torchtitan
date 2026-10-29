# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tensor-pytree contracts for linear-attention CUDA-graph metadata."""

import unittest

import torch
from attn_gym.linear.context_parallel import ContextParallelRouting
from torch.utils import _pytree as pytree

from torchtitan.models.common.attention import (
    LinearAttentionMetadata,
    VarlenAttentionMetadata,
)
from torchtitan.models.common.attention.cp_linear_attention import (
    ContextParallelLinearAttentionMetadata,
)


class TestLinearAttentionMetadataPytree(unittest.TestCase):
    def test_kda_metadata_exposes_sequence_offsets_as_pytree_leaves(self):
        """Keep sequence tensors visible to graph input copying and replay."""
        offsets = torch.tensor([0, 4], dtype=torch.int32)
        metadata = LinearAttentionMetadata(
            varlen=VarlenAttentionMetadata(offsets, offsets, 4, 4),
            num_conv_history_tokens=3,
        )

        leaves, tree_spec = pytree.tree_flatten(metadata)
        self.assertIs(leaves[0], offsets)
        self.assertIs(leaves[1], offsets)
        self.assertEqual(leaves[2:], [4, 4, -1, 0, 3])
        rebuilt = pytree.tree_unflatten(leaves, tree_spec)
        self.assertIsInstance(rebuilt, LinearAttentionMetadata)
        self.assertIs(rebuilt.varlen.cu_seq_q, offsets)
        self.assertIs(rebuilt.varlen.cu_seq_k, offsets)
        self.assertEqual(rebuilt.varlen.window_size, (-1, 0))
        self.assertEqual(rebuilt.num_conv_history_tokens, 3)

    def test_cp_kda_metadata_exposes_routing_tensors_as_pytree_leaves(self):
        """Keep every CP routing tensor visible without changing its ownership."""
        offsets = torch.tensor([0, 4], dtype=torch.int32)
        routing = ContextParallelRouting.from_fragments(
            cu_seqlens_global=[0, 4],
            fragments=[[(0, 2)], [(2, 4)]],
            cp_rank=0,
            device="cpu",
            conv_history=3,
        )
        metadata = ContextParallelLinearAttentionMetadata(
            varlen=VarlenAttentionMetadata(offsets, offsets, 4, 4),
            num_conv_history_tokens=3,
            cp_routing=routing,
        )

        leaves, tree_spec = pytree.tree_flatten(metadata)

        self.assertNotIn(metadata, leaves)
        self.assertNotIn(routing, leaves)
        for value in vars(routing).values():
            if isinstance(value, torch.Tensor):
                self.assertTrue(any(value is leaf for leaf in leaves))

        rebuilt = pytree.tree_unflatten(leaves, tree_spec)
        self.assertIsInstance(rebuilt, ContextParallelLinearAttentionMetadata)
        self.assertIsInstance(rebuilt.cp_routing, ContextParallelRouting)
        self.assertIs(rebuilt.varlen.cu_seq_q, offsets)
        for name, value in vars(routing).items():
            rebuilt_value = getattr(rebuilt.cp_routing, name)
            if isinstance(value, torch.Tensor):
                self.assertIs(rebuilt_value, value)
            else:
                self.assertEqual(rebuilt_value, value)


if __name__ == "__main__":
    unittest.main()
