# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import patch

import pytest
import spmd_types as spmd
import torch

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.common.activation import ClampedSwiGLU, Sigmoid, SwiGLU
from torchtitan.models.common.linear.hi_mid_lo_linear import HiMidLoLinear
from torchtitan.models.common.moe import MoE
from torchtitan.models.deepseek_v4 import build_model_config
from torchtitan.models.deepseek_v4.moe import DeepSeekV4Router


@pytest.mark.parametrize(
    "flavor", ["debugmodel", "deepseek_v4_flash", "deepseek_v4_pro"]
)
def test_deepseek_v4_flavors_configure_routed_shared_and_mtp_activations(flavor):
    config = build_model_config(flavor, parallelism=ParallelismConfig(), n_mtp_layers=1)
    assert config.mtp_layers
    moe_configs = [moe for _, moe, _, _ in config.traverse(MoE.Config)]
    assert moe_configs
    for moe in moe_configs:
        activations = [moe.routed_experts.activation_fn]
        if moe.shared_experts is not None:
            activations.append(moe.shared_experts.activation_fn)
        for activation in activations:
            if flavor == "debugmodel":
                assert isinstance(activation, SwiGLU.Config)
            else:
                assert isinstance(activation, ClampedSwiGLU.Config)
                assert activation.swiglu_alpha == 1.0
                assert activation.swiglu_limit == 10.0
                assert activation.swiglu_offset == 0.0


class TestDeepSeekV4Router(unittest.TestCase):
    def test_hash_router_explicitly_shards_input_ids_across_tp(self):
        router = DeepSeekV4Router.Config(
            num_experts=4,
            gate=HiMidLoLinear.Config(in_features=4, out_features=4),
            score_func=Sigmoid.Config(),
            top_k=1,
            vocab_size=8,
            n_hash_layers=1,
            layer_id=0,
        ).build()
        scores_TE = torch.randn(4, 4)
        input_ids_T = torch.arange(4)
        tp_group = object()

        with (
            patch(
                "torchtitan.models.deepseek_v4.moe.spmd_sparse_mesh",
                return_value=object(),
            ),
            patch(
                "torchtitan.models.deepseek_v4.moe.spmd_mesh_group",
                return_value=tp_group,
            ),
            patch(
                "torchtitan.models.deepseek_v4.moe.spmd.redistribute",
                side_effect=lambda tensor, *_args, **_kwargs: tensor,
            ) as redistribute,
        ):
            actual_expert_ids_T1 = router._select_experts(
                scores_TE,
                input_ids_T=input_ids_T,
            )

        torch.testing.assert_close(
            actual_expert_ids_T1,
            router.tid2eid[input_ids_T],
        )
        redistribute.assert_called_once_with(
            input_ids_T,
            tp_group,
            src=spmd.R,
            dst=spmd.S(0),
            backward_options={"op_dtype": input_ids_T.dtype},
        )


if __name__ == "__main__":
    unittest.main()
