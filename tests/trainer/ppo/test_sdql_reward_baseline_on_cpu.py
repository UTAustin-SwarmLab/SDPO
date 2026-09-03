# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CPU checks for SDQL ``use_reward_baseline`` (leave-one-out reward centering)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from verl.trainer.ppo.core_algos import (
    compute_self_distillation_q_loss,
    masked_leave_one_out_baseline,
)


def _sdql_cfg(**overrides):
    cfg = dict(
        full_logit_distillation=True,
        distillation_topk=None,
        distillation_add_tail=False,
        alpha=0.0,
        gamma=1.0,
        is_clip=None,
        use_reward_clamp=False,
        use_reward_baseline=True,
        use_env_reward=False,
        env_reward_scale=1.0,
        target_q_mode="uniform",
        clamp_low=-10.0,
        clamp_high=10.0,
    )
    cfg.update(overrides)
    return SimpleNamespace(**cfg)


class TestMaskedLeaveOneOutBroadcast:
    def test_broadcasts_over_action_dim(self):
        values = torch.tensor(
            [[[1.0, 2.0], [3.0, 4.0]], [[10.0, 20.0], [30.0, 40.0]]],
            dtype=torch.float64,
        )  # [B=2, T=2, K=2]
        mask = torch.ones(2, 2, dtype=torch.float64)
        baseline = masked_leave_one_out_baseline(values, mask)
        # Each row sees only the other row's values at that (t, k).
        assert torch.allclose(baseline[0], values[1])
        assert torch.allclose(baseline[1], values[0])


class TestSdqlRewardBaseline:
    def test_sampled_path_emits_metric_and_changes_loss(self):
        torch.manual_seed(0)
        B, T, V = 4, 5, 6
        s_all = F.log_softmax(torch.randn(B, T, V, dtype=torch.float64), -1)
        t_tok = torch.randn(B, T, dtype=torch.float64).clamp(max=0.0) - 1.0
        old_all = F.log_softmax(torch.randn(B, T, V, dtype=torch.float64), -1)
        actions = torch.randint(0, V, (B, T))
        s_tok = s_all.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        old_tok = old_all.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        mask = torch.ones(B, T, dtype=torch.float64)

        def run(use_reward_baseline: bool):
            return compute_self_distillation_q_loss(
                student_log_probs=s_tok,
                teacher_log_probs=t_tok,
                response_mask=mask,
                self_distillation_config=_sdql_cfg(
                    full_logit_distillation=False,
                    alpha=1.0,
                    use_reward_baseline=use_reward_baseline,
                ),
                old_log_probs=old_tok,
                old_all_log_probs=old_all,
                student_all_log_probs=s_all,
                loss_agg_mode="token-mean",
                index=np.array(["p0"] * B, dtype=object),
            )

        loss_on, metrics_on = run(True)
        loss_off, metrics_off = run(False)
        assert "sdql/reward_baseline" in metrics_on
        assert "sdql/reward_baseline" not in metrics_off
        assert torch.isfinite(loss_on)
        assert not torch.allclose(loss_on, loss_off)

    def test_full_logit_path_emits_metric(self):
        torch.manual_seed(1)
        B, T, K = 3, 4, 5
        s_topk = F.log_softmax(torch.randn(B, T, K, dtype=torch.float64), -1)
        t_topk = F.log_softmax(torch.randn(B, T, K, dtype=torch.float64), -1)
        old_topk = F.log_softmax(torch.randn(B, T, K, dtype=torch.float64), -1)
        s_tok = s_topk[..., 0]
        t_tok = t_topk[..., 0]
        old_tok = old_topk[..., 0]
        mask = torch.ones(B, T, dtype=torch.float64)

        loss, metrics = compute_self_distillation_q_loss(
            student_log_probs=s_tok,
            teacher_log_probs=t_tok,
            response_mask=mask,
            self_distillation_config=_sdql_cfg(
                use_reward_baseline=True,
                alpha=0.0,
                distillation_topk=K,
                distillation_add_tail=False,
            ),
            old_log_probs=old_tok,
            old_topk_log_probs=old_topk,
            student_topk_log_probs=s_topk,
            teacher_topk_log_probs=t_topk,
            loss_agg_mode="token-mean",
            index=np.array(["p0"] * B, dtype=object),
        )
        assert torch.isfinite(loss)
        assert "sdql/reward_baseline" in metrics


class TestSdqlFullLogitLambdaReturns:
    def test_uses_expected_q_value_and_sampled_future_rewards(self):
        dtype = torch.float64
        student = torch.log(
            torch.tensor(
                [[[0.7, 0.3], [0.4, 0.6], [0.8, 0.2]]],
                dtype=dtype,
            )
        )
        teacher = torch.log(
            torch.tensor(
                [[[0.6, 0.4], [0.5, 0.5], [0.3, 0.7]]],
                dtype=dtype,
            )
        )
        old = torch.log(
            torch.tensor(
                [[[0.5, 0.5], [0.7, 0.3], [0.6, 0.4]]],
                dtype=dtype,
            )
        )
        sampled_token_ids = torch.tensor([[10, 20, 30]])
        topk_indices = torch.tensor([[[10, 11], [21, 20], [30, 31]]])
        sampled_positions = torch.tensor([[0, 1, 0]])
        sampled_student = student.gather(-1, sampled_positions.unsqueeze(-1)).squeeze(-1)
        sampled_teacher = teacher.gather(-1, sampled_positions.unsqueeze(-1)).squeeze(-1)
        sampled_old = old.gather(-1, sampled_positions.unsqueeze(-1)).squeeze(-1)
        mask = torch.ones(1, 3, dtype=dtype)
        gamma = 0.9
        lambda_ = 0.5

        loss, _ = compute_self_distillation_q_loss(
            student_log_probs=sampled_student,
            teacher_log_probs=sampled_teacher,
            response_mask=mask,
            self_distillation_config=_sdql_cfg(
                alpha=1.0,
                distillation_topk=2,
                target_q_mode="on-policy-lambda",
                use_reward_baseline=False,
                gamma=gamma,
                lambda_=lambda_,
            ),
            old_log_probs=sampled_old,
            old_topk_log_probs=old,
            sampled_token_ids=sampled_token_ids,
            student_topk_indices=topk_indices,
            student_topk_log_probs=student,
            teacher_topk_log_probs=teacher,
            loss_agg_mode="token-mean",
        )

        q_values = student - old
        rewards = teacher - student
        values = (student.exp() * q_values).sum(dim=-1)
        sampled_rewards = rewards.gather(-1, sampled_positions.unsqueeze(-1)).squeeze(-1)

        expected_targets = torch.empty_like(rewards)
        sampled_return = torch.zeros(1, dtype=dtype)
        for t in reversed(range(rewards.size(1))):
            if t + 1 < rewards.size(1):
                continuation = (1.0 - lambda_) * values[:, t + 1] + lambda_ * sampled_return
            else:
                continuation = torch.zeros_like(sampled_return)
            expected_targets[:, t, :] = rewards[:, t, :] + gamma * continuation.unsqueeze(-1)
            sampled_return = sampled_rewards[:, t] + gamma * continuation

        expected_loss = ((q_values - expected_targets) ** 2).sum(dim=-1).mean()
        assert torch.allclose(loss, expected_loss)
