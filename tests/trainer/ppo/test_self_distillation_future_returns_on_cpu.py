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
"""Gradient / padding checks for self-distillation ``use_future_returns``.

Ported from the /tmp verify_gae_distill / verify_gae_sampled / verify_vs_distil /
verify_padding scripts. Uses a tiny tabular autoregressive model so the
on-policy objective can be enumerated exactly.
"""

from __future__ import annotations

import itertools
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from verl.trainer.ppo.core_algos import (
    compute_distil_self_distillation_loss,
    compute_self_distillation_loss,
    discounted_cumsum,
    masked_leave_one_out_baseline,
)


def _base_cfg(**overrides):
    cfg = dict(
        full_logit_distillation=True,
        distillation_topk=None,
        distillation_add_tail=False,
        alpha=0.0,
        use_future_returns=True,
        # Exact single-sequence gradient checks compare against the raw future return.
        use_future_returns_baseline=False,
        gamma=1.0,
        is_clip=None,
        use_reward_clamp=False,
        clamp_low=-10.0,
        clamp_high=10.0,
    )
    cfg.update(overrides)
    return SimpleNamespace(**cfg)


def _tabular_model(V: int, T: int, seed: int = 0):
    torch.manual_seed(seed)
    prefixes = []
    for t in range(T):
        prefixes.extend(itertools.product(range(V), repeat=t))
    pidx = {p: i for i, p in enumerate(prefixes)}
    student_logits = torch.randn(len(prefixes), V, dtype=torch.float64, requires_grad=True)
    teacher_logits = torch.randn(len(prefixes), V, dtype=torch.float64)

    def student_logp(prefix):
        return F.log_softmax(student_logits[pidx[prefix]], dim=-1)

    def teacher_logp(prefix):
        return F.log_softmax(teacher_logits[pidx[prefix]], dim=-1)

    return student_logits, student_logp, teacher_logp


def _grad(x, params):
    return torch.autograd.grad(x, params, retain_graph=True)[0]


def _enumerate_code_objective(loss_fn, cfg, student_logp, teacher_logp, V, T, full_logit: bool):
    total = 0.0
    for seq in itertools.product(range(V), repeat=T):
        s_all, t_all, s_tok, t_tok = [], [], [], []
        logp_seq = 0.0
        for t in range(T):
            prefix = seq[:t]
            sl, tl = student_logp(prefix), teacher_logp(prefix)
            if full_logit:
                s_all.append(sl)
                t_all.append(tl)
            s_tok.append(sl[seq[t]])
            t_tok.append(tl[seq[t]])
            logp_seq = logp_seq + sl[seq[t]].detach()
        kwargs = dict(
            student_log_probs=torch.stack(s_tok).unsqueeze(0),
            teacher_log_probs=torch.stack(t_tok).unsqueeze(0),
            response_mask=torch.ones(1, T, dtype=torch.float64),
            self_distillation_config=cfg,
            loss_agg_mode="seq-mean-token-sum",
        )
        if full_logit:
            kwargs["student_all_log_probs"] = torch.stack(s_all).unsqueeze(0)
            kwargs["teacher_all_log_probs"] = torch.stack(t_all).unsqueeze(0)
        loss, _ = loss_fn(**kwargs)
        total = total + logp_seq.exp() * loss
    return total


class TestFutureReturnsFullLogitGradient:
    """Full-logit path: grad of surrogate == exact total derivative at gamma=1."""

    @pytest.mark.parametrize("gamma", [1.0, 0.99, 0.9])
    def test_matches_exact_forward_kl_grad(self, gamma):
        V, T = 3, 3
        student_logits, student_logp, teacher_logp = _tabular_model(V, T)

        def exact_objective():
            total = 0.0
            for t in range(T):
                for prefix in itertools.product(range(V), repeat=t):
                    reach = torch.tensor(1.0, dtype=torch.float64)
                    for i in range(t):
                        reach = reach * student_logp(prefix[:i])[prefix[i]].exp()
                    tl, sl = teacher_logp(prefix), student_logp(prefix)
                    d = (tl.exp() * (tl - sl)).sum()
                    total = total + reach * d
            return total

        g_exact = _grad(exact_objective(), student_logits)
        cfg = _base_cfg(full_logit_distillation=True, alpha=0.0, use_future_returns=True, gamma=gamma)
        g_code = _grad(
            _enumerate_code_objective(
                compute_self_distillation_loss, cfg, student_logp, teacher_logp, V, T, full_logit=True
            ),
            student_logits,
        )

        if gamma == 1.0:
            rel = (g_code - g_exact).norm() / g_exact.norm()
            assert rel < 1e-6
        cos = F.cosine_similarity(g_code.flatten(), g_exact.flatten(), dim=0)
        # Discounted returns are a biased proxy for the undiscounted exact objective;
        # still require strong directional agreement for mild discounting.
        assert cos > (0.999 if gamma == 1.0 else 0.95)


class TestFutureReturnsSampledGradient:
    """Sampled-token path (alpha=1 reverse-KL style reward) with future returns."""

    def test_matches_exact_reverse_kl_grad_gamma_one(self):
        V, T = 3, 3
        student_logits, student_logp, teacher_logp = _tabular_model(V, T)

        def exact_reverse_kl():
            total = 0.0
            for t in range(T):
                for prefix in itertools.product(range(V), repeat=t):
                    reach = torch.tensor(1.0, dtype=torch.float64)
                    for i in range(t):
                        reach = reach * student_logp(prefix[:i])[prefix[i]].exp()
                    sl, tl = student_logp(prefix), teacher_logp(prefix)
                    total = total + reach * (sl.exp() * (sl - tl)).sum()
            return total

        g_exact = _grad(exact_reverse_kl(), student_logits)
        cfg = _base_cfg(full_logit_distillation=False, alpha=1.0, use_future_returns=True, gamma=1.0)
        g_code = _grad(
            _enumerate_code_objective(
                compute_self_distillation_loss, cfg, student_logp, teacher_logp, V, T, full_logit=False
            ),
            student_logits,
        )
        rel = (g_code - g_exact).norm() / g_exact.norm()
        cos = F.cosine_similarity(g_code.flatten(), g_exact.flatten(), dim=0)
        assert rel < 1e-5
        assert cos > 0.999


class TestFutureReturnsVsDistil:
    """DISTIL (/remaining) is a biased proxy; raw future returns match exact."""

    def test_distil_diverges_future_returns_matches(self):
        V, T = 3, 4
        student_logits, student_logp, teacher_logp = _tabular_model(V, T)

        def exact_forward_kl():
            total = 0.0
            for t in range(T):
                for prefix in itertools.product(range(V), repeat=t):
                    reach = torch.tensor(1.0, dtype=torch.float64)
                    for i in range(t):
                        reach = reach * student_logp(prefix[:i])[prefix[i]].exp()
                    tl, sl = teacher_logp(prefix), student_logp(prefix)
                    total = total + reach * (tl.exp() * (tl - sl)).sum()
            return total

        g_exact = _grad(exact_forward_kl(), student_logits)
        base = dict(
            full_logit_distillation=True,
            distillation_topk=None,
            distillation_add_tail=False,
            alpha=0.0,
            is_clip=None,
            use_reward_clamp=False,
            clamp_low=-10.0,
            clamp_high=10.0,
            use_future_returns=True,
            gamma=1.0,
            use_future_returns_baseline=False,
        )

        g_fr = _grad(
            _enumerate_code_objective(
                compute_self_distillation_loss,
                SimpleNamespace(**base),
                student_logp,
                teacher_logp,
                V,
                T,
                full_logit=True,
            ),
            student_logits,
        )
        g_distil = _grad(
            _enumerate_code_objective(
                compute_distil_self_distillation_loss,
                SimpleNamespace(**base),
                student_logp,
                teacher_logp,
                V,
                T,
                full_logit=True,
            ),
            student_logits,
        )
        g_nopg = _grad(
            _enumerate_code_objective(
                compute_self_distillation_loss,
                SimpleNamespace(**{**base, "use_future_returns": False}),
                student_logp,
                teacher_logp,
                V,
                T,
                full_logit=True,
            ),
            student_logits,
        )

        cos_fr = F.cosine_similarity(g_fr.flatten(), g_exact.flatten(), dim=0).item()
        cos_distil = F.cosine_similarity(g_distil.flatten(), g_exact.flatten(), dim=0).item()
        cos_nopg = F.cosine_similarity(g_nopg.flatten(), g_exact.flatten(), dim=0).item()

        assert cos_fr > 0.999
        assert (g_fr - g_exact).norm() / g_exact.norm() < 1e-5
        # DISTIL's /remaining reweighting is a biased proxy of the exact PG term.
        assert cos_distil < cos_fr
        assert cos_distil < 0.995
        assert cos_nopg < cos_fr


class TestFutureReturnsPadding:
    def test_loss_padding_invariant(self):
        torch.manual_seed(0)
        V, T_real, T_pad = 5, 4, 7
        cfg = _base_cfg(use_future_returns=True, gamma=0.95)

        s_all = F.log_softmax(torch.randn(1, T_pad, V, dtype=torch.float64), -1)
        t_all = F.log_softmax(torch.randn(1, T_pad, V, dtype=torch.float64), -1)
        s_tok, t_tok = s_all[..., 0], t_all[..., 0]

        loss_short, _ = compute_self_distillation_loss(
            student_log_probs=s_tok[:, :T_real],
            teacher_log_probs=t_tok[:, :T_real],
            response_mask=torch.ones(1, T_real, dtype=torch.float64),
            self_distillation_config=cfg,
            student_all_log_probs=s_all[:, :T_real],
            teacher_all_log_probs=t_all[:, :T_real],
            loss_agg_mode="seq-mean-token-sum",
        )

        mask = torch.zeros(1, T_pad, dtype=torch.float64)
        mask[:, :T_real] = 1.0
        loss_pad, _ = compute_self_distillation_loss(
            student_log_probs=s_tok,
            teacher_log_probs=t_tok,
            response_mask=mask,
            self_distillation_config=cfg,
            student_all_log_probs=s_all,
            teacher_all_log_probs=t_all,
            loss_agg_mode="seq-mean-token-sum",
        )
        assert torch.allclose(loss_short, loss_pad)


class TestDiscountedCumsumConvention:
    def test_shifted_future_return_matches_formula(self):
        r = torch.tensor([[1.0, 2.0, 4.0, 8.0]], dtype=torch.float64)
        gamma = 0.5
        shifted = torch.cat([r[:, 1:], torch.zeros_like(r[:, :1])], dim=1)
        G = torch.flip(discounted_cumsum(torch.flip(shifted, dims=[1]), gamma), dims=[1])
        expected = [
            sum(gamma ** (k - t - 1) * r[0, k].item() for k in range(t + 1, 4)) for t in range(4)
        ]
        assert torch.allclose(G[0], torch.tensor(expected, dtype=torch.float64))


class TestFutureReturnsBaseline:
    def test_leave_one_out_excludes_self_and_padding(self):
        values = torch.tensor(
            [[1.0, 2.0, 3.0], [10.0, 20.0, 30.0], [100.0, 200.0, 300.0]], dtype=torch.float64
        )
        mask = torch.tensor(
            [[1.0, 1.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]], dtype=torch.float64
        )
        baseline = masked_leave_one_out_baseline(values, mask)
        # column 0: all three valid -> mean of the other two
        assert torch.allclose(
            baseline[:, 0], torch.tensor([55.0, 50.5, 5.5], dtype=torch.float64)
        )
        # column 1: rows 0 and 2 valid -> each sees only the other
        assert baseline[0, 1].item() == pytest.approx(200.0)
        assert baseline[2, 1].item() == pytest.approx(2.0)
        # column 2: nothing valid -> zero baseline
        assert torch.allclose(baseline[:, 2], torch.zeros(3, dtype=torch.float64))

    def test_single_valid_row_gets_zero_baseline(self):
        values = torch.tensor([[5.0], [7.0]], dtype=torch.float64)
        mask = torch.tensor([[1.0], [0.0]], dtype=torch.float64)
        baseline = masked_leave_one_out_baseline(values, mask)
        assert baseline[0, 0].item() == pytest.approx(0.0)

    def test_batch_of_one_keeps_the_policy_gradient_term(self):
        """Self-inclusive means would cancel the PG term entirely at B=1."""
        torch.manual_seed(3)
        T, V = 5, 4
        s_all = F.log_softmax(torch.randn(1, T, V, dtype=torch.float64), -1)
        t_all = F.log_softmax(torch.randn(1, T, V, dtype=torch.float64), -1)
        actions = torch.randint(0, V, (1, T))
        s_tok = s_all.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        t_tok = t_all.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        mask = torch.ones(1, T, dtype=torch.float64)

        def run(**over):
            loss, _ = compute_self_distillation_loss(
                student_log_probs=s_tok,
                teacher_log_probs=t_tok,
                response_mask=mask,
                self_distillation_config=_base_cfg(**over),
                student_all_log_probs=s_all,
                teacher_all_log_probs=t_all,
                loss_agg_mode="seq-mean-token-sum",
            )
            return loss

        with_baseline = run(use_future_returns=True, use_future_returns_baseline=True)
        without_pg = run(use_future_returns=False)
        raw = run(use_future_returns=True, use_future_returns_baseline=False)
        assert torch.allclose(with_baseline, raw)
        assert not torch.allclose(with_baseline, without_pg)

    def test_full_logit_baseline_reduces_return_magnitude(self):
        torch.manual_seed(0)
        B, T, V = 8, 5, 4
        cfg = _base_cfg(use_future_returns=True, use_future_returns_baseline=True, gamma=1.0)
        s_all = F.log_softmax(torch.randn(B, T, V, dtype=torch.float64), -1)
        t_all = F.log_softmax(torch.randn(B, T, V, dtype=torch.float64), -1)
        actions = torch.randint(0, V, (B, T))
        s_tok = s_all.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        t_tok = t_all.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        mask = torch.ones(B, T, dtype=torch.float64)
        mask[:, -1] = 0.0  # pad last token on all seqs

        with torch.no_grad():
            kl = F.kl_div(s_all, t_all, reduction="none", log_target=True).sum(-1)
            reward = kl * mask
            shifted = torch.cat([reward[:, 1:], torch.zeros_like(reward[:, :1])], dim=1)
            G = torch.flip(discounted_cumsum(torch.flip(shifted, dims=[1]), 1.0), dims=[1])
            centered = G - masked_leave_one_out_baseline(G, mask)
            assert (centered * mask).abs().sum() < (G * mask).abs().sum()

        loss, metrics = compute_self_distillation_loss(
            student_log_probs=s_tok,
            teacher_log_probs=t_tok,
            response_mask=mask,
            self_distillation_config=cfg,
            student_all_log_probs=s_all,
            teacher_all_log_probs=t_all,
            loss_agg_mode="seq-mean-token-sum",
        )
        assert torch.isfinite(loss)
        assert "self_distillation/future_returns_baseline" in metrics

    def test_sampled_logit_baseline_runs(self):
        torch.manual_seed(1)
        B, T = 6, 4
        cfg = _base_cfg(
            full_logit_distillation=False,
            alpha=1.0,
            use_future_returns=True,
            use_future_returns_baseline=True,
            gamma=1.0,
            use_reward_clamp=False,
        )
        s_tok = torch.randn(B, T, dtype=torch.float64).clamp(max=0.0) - 1.0
        t_tok = torch.randn(B, T, dtype=torch.float64).clamp(max=0.0) - 1.0
        mask = torch.ones(B, T, dtype=torch.float64)
        loss, metrics = compute_self_distillation_loss(
            student_log_probs=s_tok,
            teacher_log_probs=t_tok,
            response_mask=mask,
            self_distillation_config=cfg,
            loss_agg_mode="token-mean",
        )
        assert torch.isfinite(loss)
        assert "self_distillation/future_returns_baseline" in metrics


class TestFutureReturnsCispoSplit:
    """Full-logit + future returns: one-sided IS on KL, CISPO on the PG term."""

    def test_split_differs_from_joint_onesided_and_matches_manual(self):
        torch.manual_seed(0)
        B, T, V = 2, 4, 5
        is_clip = 0.2
        s_all = F.log_softmax(torch.randn(B, T, V, dtype=torch.float64), -1)
        t_all = F.log_softmax(torch.randn(B, T, V, dtype=torch.float64), -1)
        old_all = F.log_softmax(torch.randn(B, T, V, dtype=torch.float64), -1)
        actions = torch.randint(0, V, (B, T))
        s_tok = s_all.gather(-1, actions.unsqueeze(-1)).squeeze(-1).requires_grad_(True)
        t_tok = t_all.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        old_tok = old_all.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        mask = torch.ones(B, T, dtype=torch.float64)

        cfg = _base_cfg(
            use_future_returns=True,
            use_future_returns_baseline=False,
            gamma=1.0,
            is_clip=is_clip,
        )
        loss, metrics = compute_self_distillation_loss(
            student_log_probs=s_tok,
            teacher_log_probs=t_tok,
            response_mask=mask,
            self_distillation_config=cfg,
            old_log_probs=old_tok,
            student_all_log_probs=s_all,
            teacher_all_log_probs=t_all,
            loss_agg_mode="token-mean",
        )
        assert "self_distillation/future_returns_cispo_clipfrac" in metrics

        with torch.no_grad():
            kl = F.kl_div(s_all, t_all, reduction="none", log_target=True).sum(-1)
            reward = kl * mask
            shifted = torch.cat([reward[:, 1:], torch.zeros_like(reward[:, :1])], dim=1)
            G = torch.flip(discounted_cumsum(torch.flip(shifted, dims=[1]), 1.0), dims=[1])
            ratio = torch.exp((s_tok.detach() - old_tok).clamp(-20.0, 20.0))
            distill_ratio = ratio.clamp(max=is_clip)
            cispo_ratio = ratio.clamp(min=1.0 - is_clip, max=1.0 + is_clip)
            expected = (kl * distill_ratio + G * s_tok * cispo_ratio) * mask
            expected_loss = expected.sum() / mask.sum()

            # Old joint one-sided weighting of (KL + PG) should differ once any ratio is clipped.
            joint_onesided = ((kl + G * s_tok.detach()) * distill_ratio * mask).sum() / mask.sum()

        assert torch.allclose(loss, expected_loss)
        assert not torch.allclose(loss.detach(), joint_onesided)

    def test_no_future_returns_keeps_onesided_is(self):
        torch.manual_seed(1)
        B, T, V = 2, 3, 4
        is_clip = 0.5
        s_all = F.log_softmax(torch.randn(B, T, V, dtype=torch.float64), -1)
        t_all = F.log_softmax(torch.randn(B, T, V, dtype=torch.float64), -1)
        old_all = F.log_softmax(torch.randn(B, T, V, dtype=torch.float64), -1)
        actions = torch.randint(0, V, (B, T))
        s_tok = s_all.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        t_tok = t_all.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        old_tok = old_all.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        mask = torch.ones(B, T, dtype=torch.float64)

        loss, metrics = compute_self_distillation_loss(
            student_log_probs=s_tok,
            teacher_log_probs=t_tok,
            response_mask=mask,
            self_distillation_config=_base_cfg(use_future_returns=False, is_clip=is_clip),
            old_log_probs=old_tok,
            student_all_log_probs=s_all,
            teacher_all_log_probs=t_all,
            loss_agg_mode="token-mean",
        )
        assert "self_distillation/future_returns_cispo_clipfrac" not in metrics

        with torch.no_grad():
            kl = F.kl_div(s_all, t_all, reduction="none", log_target=True).sum(-1)
            ratio = torch.exp((s_tok - old_tok).clamp(-20.0, 20.0)).clamp(max=is_clip)
            expected = (kl * ratio * mask).sum() / mask.sum()
        assert torch.allclose(loss, expected)
