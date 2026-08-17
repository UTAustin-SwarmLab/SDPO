#!/usr/bin/env python3
"""Equal- vs unequal-length credit-weight estimator study.

Compares:
  raw G_t
  G_t / (T-1)
  G_t - mean_t(G_t)
  (G_t - mean_t(G_t)) / (T-1)
  center r_t then G   [i.e. G(r - mean_t(r))]
  no PG term

Under:
  equal lengths   — every trajectory has length T
  unequal lengths — right-padded; L ~ Uniform{T_min..T}, independent of the chain

Exact grad accounts for survival P(L > t) when lengths vary.

Run from repo root:
  python tests/special_standalone/future_returns_variance/variance_equal_vs_unequal.py
  python tests/special_standalone/future_returns_variance/variance_equal_vs_unequal.py \\
      --T 64 256 --N 10000 --batch-size 16
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def exact_grad(
    theta: torch.Tensor,
    teacher: torch.Tensor,
    teacher_lp: torch.Tensor,
    T: int,
    survival: torch.Tensor | None = None,
) -> torch.Tensor:
    """∇J with J = sum_t survival[t] * E_{s~P_t}[KL(teacher||pi)].

    survival[t] = P(L > t). None → all ones (fixed length T).
    """
    pi = F.softmax(theta, -1)
    d = (teacher * (teacher_lp - F.log_softmax(theta, -1))).sum(-1)
    P = torch.full((theta.shape[0],), 1.0 / theta.shape[0], dtype=theta.dtype)
    if survival is None:
        survival = torch.ones(T, dtype=theta.dtype)
    J = theta.new_zeros(())
    for t in range(T):
        J = J + survival[t] * (P * d).sum()
        P = (P.unsqueeze(-1) * pi).sum(0)
        P = P / P.sum()
    return torch.autograd.grad(J, theta)[0]


@torch.no_grad()
def rollout(theta, teacher, teacher_lp, T: int, N: int, lengths: torch.Tensor | None = None):
    """Roll out N trajectories of pad length T.

    lengths: [N] int in 1..T. None → all length T.
    Returns states, actions, rewards, mask  all [N, T].
    """
    V = theta.shape[0]
    pi = F.softmax(theta, -1)
    d = (teacher * (teacher_lp - F.log_softmax(theta, -1))).sum(-1)
    s = torch.randint(0, V, (N,))
    states, actions = [], []
    for _ in range(T):
        a = torch.multinomial(pi[s], 1).squeeze(-1)
        states.append(s)
        actions.append(a)
        s = a
    states = torch.stack(states, 1)
    actions = torch.stack(actions, 1)
    rewards = d[states]
    if lengths is None:
        mask = torch.ones(N, T, dtype=rewards.dtype)
    else:
        t = torch.arange(T).unsqueeze(0)
        mask = (t < lengths.unsqueeze(1)).to(rewards.dtype)
        rewards = rewards * mask
    return states, actions, rewards, mask


def future_returns(rewards: torch.Tensor) -> torch.Tensor:
    """G_t = sum_{k>t} r_k  (gamma=1, shifted)."""
    shifted = torch.cat([rewards[:, 1:], torch.zeros_like(rewards[:, :1])], 1)
    return torch.flip(torch.cumsum(torch.flip(shifted, [1]), 1), [1])


def masked_batch_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Position-wise mean over batch, [1, T]."""
    denom = mask.sum(dim=0, keepdim=True).clamp_min(1.0)
    return (values * mask).sum(dim=0, keepdim=True) / denom


@torch.no_grad()
def per_sample_grads(theta, teacher, states, actions, rewards, mask, weight: torch.Tensor):
    """Analytic per-sample grads [N,V,V] with credit weights `weight` [N,T]."""
    V = theta.shape[0]
    N, T = rewards.shape
    pi = F.softmax(theta, -1)
    onehot = F.one_hot(actions, V).to(rewards.dtype)
    pi_s = pi[states]
    # Direct KL term only on valid tokens; PG term uses weight (already masked by caller).
    direct = (pi_s - teacher[states]) * mask.unsqueeze(-1)
    pg = weight.unsqueeze(-1) * (onehot - pi_s) * mask.unsqueeze(-1)
    contrib = direct + pg
    g = torch.zeros(N, V, V, dtype=rewards.dtype)
    g.scatter_add_(1, states.unsqueeze(-1).expand(-1, -1, V), contrib)
    return g


def make_weights(name: str, rewards: torch.Tensor, mask: torch.Tensor, T: int) -> torch.Tensor:
    """Build credit weight W [N,T] for a named estimator."""
    G = future_returns(rewards)
    if name == "raw G_t":
        W = G
    elif name == "G_t / (T-1)":
        W = G / (T - 1)
    elif name == "G_t - mean_t(G_t)":
        W = G - masked_batch_mean(G, mask)
    elif name == "(G_t - mean_t(G_t))/(T-1)":
        W = (G - masked_batch_mean(G, mask)) / (T - 1)
    elif name == "G(r_t - mean_t(r_t))":
        r_c = rewards - masked_batch_mean(rewards, mask)
        r_c = r_c * mask
        W = future_returns(r_c)
    elif name == "no PG term":
        W = torch.zeros_like(G)
    else:
        raise ValueError(name)
    return W * mask


VARIANTS = [
    "raw G_t",
    "G_t / (T-1)",
    "G_t - mean_t(G_t)",
    "(G_t - mean_t(G_t))/(T-1)",
    "G(r_t - mean_t(r_t))",
    "no PG term",
]


def metrics(g: torch.Tensor, g_star: torch.Tensor) -> dict:
    mean = g.mean(0)
    noise = (g - mean).pow(2).sum(dim=(1, 2)).mean().sqrt()
    snr = (mean.norm() / noise).item() if noise > 0 else float("nan")
    cos = F.cosine_similarity(mean.flatten(), g_star.flatten(), dim=0).item()
    return {
        "snr": snr,
        "cos": cos,
        "mean_norm": mean.norm().item(),
        "noise": noise.item(),
    }


def run_setting(
    label: str,
    theta,
    teacher,
    teacher_lp,
    T: int,
    N: int,
    batch_size: int,
    lengths: torch.Tensor | None,
    survival: torch.Tensor | None,
    reps: int,
):
    g_star = exact_grad(theta, teacher, teacher_lp, T, survival=survival)
    st, ac, rw, mask = rollout(theta, teacher, teacher_lp, T, N, lengths=lengths)

    print(f"\n=== {label}  T={T}  N={N}  micro-B={batch_size}  "
          f"||exact||={g_star.norm():.3f}  mean_r={rw.mean():.4f}  "
          f"mean_len={mask.sum(1).mean():.1f} ===")

    rows = []
    # Full-pool estimate (large N) — like the original study
    print("  -- full pool (batch mean over all N) --")
    for name in VARIANTS:
        W = make_weights(name, rw, mask, T)
        g = per_sample_grads(theta, teacher, st, ac, rw, mask, W)
        m = metrics(g, g_star)
        print(f"  {name:<32} SNR={m['snr']:9.4f}  cos={m['cos']:.5f}  "
              f"||mean||={m['mean_norm']:.3f}  noise={m['noise']:.3f}")
        rows.append({"setting": label, "scope": "full_pool", "T": T, "N": N,
                     "batch_size": N, "variant": name, **m})

    # Micro-batch estimate: average metrics over random subsets of size batch_size
    # (matches training: baseline computed inside each micro-batch)
    if batch_size < N:
        print(f"  -- micro-batch B={batch_size}  ({reps} random subsets) --")
        acc = {name: {"cos": [], "snr": [], "mean_norm": []} for name in VARIANTS}
        for _ in range(reps):
            idx = torch.randperm(N)[:batch_size]
            st_b, ac_b, rw_b, mask_b = st[idx], ac[idx], rw[idx], mask[idx]
            for name in VARIANTS:
                W = make_weights(name, rw_b, mask_b, T)
                g = per_sample_grads(theta, teacher, st_b, ac_b, rw_b, mask_b, W)
                m = metrics(g, g_star)
                acc[name]["cos"].append(m["cos"])
                acc[name]["snr"].append(m["snr"])
                acc[name]["mean_norm"].append(m["mean_norm"])
        for name in VARIANTS:
            cos = torch.tensor(acc[name]["cos"])
            snr = torch.tensor(acc[name]["snr"])
            print(f"  {name:<32} SNR={snr.mean():9.4f}±{snr.std():.3f}  "
                  f"cos={cos.mean():.5f}±{cos.std():.4f}")
            rows.append({
                "setting": label, "scope": "micro_batch", "T": T, "N": N,
                "batch_size": batch_size, "variant": name,
                "snr": snr.mean().item(), "snr_std": snr.std().item(),
                "cos": cos.mean().item(), "cos_std": cos.std().item(),
                "mean_norm": torch.tensor(acc[name]["mean_norm"]).mean().item(),
            })
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--V", type=int, default=5)
    p.add_argument("--T", type=int, nargs="+", default=[64, 256])
    p.add_argument("--N", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=16, help="micro-batch size for subset eval")
    p.add_argument("--reps", type=int, default=64, help="random micro-batch subsets")
    p.add_argument("--T-min-frac", type=float, default=0.25,
                   help="unequal: L ~ Uniform{ceil(frac*T)..T}")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    torch.set_default_dtype(torch.float64)

    theta = torch.randn(args.V, args.V, requires_grad=True)
    teacher_logits = torch.randn(args.V, args.V) * 1.5
    teacher = F.softmax(teacher_logits, -1)
    teacher_lp = F.log_softmax(teacher_logits, -1)

    all_rows = []
    for T in args.T:
        # Equal lengths
        all_rows += run_setting(
            "equal_lengths", theta, teacher, teacher_lp, T, args.N,
            args.batch_size, lengths=None, survival=None, reps=args.reps,
        )

        # Unequal lengths: L ~ Uniform{T_min..T}
        T_min = max(2, int(torch.ceil(torch.tensor(args.T_min_frac * T)).item()))
        lengths = torch.randint(T_min, T + 1, (args.N,))
        # Closed-form survival under Uniform{T_min..T}
        # P(L = ℓ) = 1/(T - T_min + 1) for ℓ in T_min..T
        n_len = T - T_min + 1
        survival = torch.zeros(T, dtype=torch.float64)
        for t in range(T):
            # P(L > t) = number of ℓ in [T_min..T] with ℓ > t, over n_len
            lo = max(T_min, t + 1)
            survival[t] = max(0, T - lo + 1) / n_len
        print(f"\n  [unequal] T_min={T_min}  E[L]={(T_min+T)/2:.1f}  "
              f"survival[0]={survival[0]:.3f}  survival[T/2]={survival[T//2]:.3f}")
        all_rows += run_setting(
            "unequal_lengths", theta, teacher, teacher_lp, T, args.N,
            args.batch_size, lengths=lengths, survival=survival, reps=args.reps,
        )

    # Compact comparison table: equal vs unequal cos at micro-batch
    print("\n" + "=" * 78)
    print("SUMMARY  (micro-batch cosine → exact; equal vs unequal)")
    print("=" * 78)
    for T in args.T:
        print(f"\nT={T}, B={args.batch_size}")
        print(f"  {'variant':<32} {'equal cos':>12} {'unequal cos':>12} {'|Δ|':>10} "
              f"{'equal SNR':>12} {'unequal SNR':>12}")
        for name in VARIANTS:
            eq = next(r for r in all_rows
                      if r["setting"] == "equal_lengths" and r["scope"] == "micro_batch"
                      and r["T"] == T and r["variant"] == name)
            uq = next(r for r in all_rows
                      if r["setting"] == "unequal_lengths" and r["scope"] == "micro_batch"
                      and r["T"] == T and r["variant"] == name)
            print(f"  {name:<32} {eq['cos']:12.5f} {uq['cos']:12.5f} "
                  f"{abs(eq['cos']-uq['cos']):10.5f} {eq['snr']:12.4f} {uq['snr']:12.4f}")

        # Equivalence check: G-mean(G) vs G(r-mean(r)) on full pool
        print(f"\n  equivalence |cos(G−meanG) − cos(G(r−meanr))| "
              f"(full pool):")
        for setting in ("equal_lengths", "unequal_lengths"):
            a = next(r for r in all_rows
                     if r["setting"] == setting and r["scope"] == "full_pool"
                     and r["T"] == T and r["variant"] == "G_t - mean_t(G_t)")
            b = next(r for r in all_rows
                     if r["setting"] == setting and r["scope"] == "full_pool"
                     and r["T"] == T and r["variant"] == "G(r_t - mean_t(r_t))")
            print(f"    {setting:<18} cos_G={a['cos']:.6f}  cos_r={b['cos']:.6f}  "
                  f"|Δ|={abs(a['cos']-b['cos']):.2e}  "
                  f"|ΔSNR|={abs(a['snr']-b['snr']):.2e}")

    if args.out:
        Path(args.out).write_text(json.dumps(all_rows, indent=2))
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
