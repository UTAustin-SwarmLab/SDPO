#!/usr/bin/env python3
"""Sweep batch size N = 2^k for future-return credit-weight estimators.

For each N, the position-wise batch baseline is computed on those same N
trajectories (as in a real training step). Metrics are averaged over random
subsets drawn from a fixed rollout pool.

Run from repo root:
    python tests/special_standalone/future_returns_variance/variance_batch_size_sweep.py
    python tests/special_standalone/future_returns_variance/variance_batch_size_sweep.py \\
        --T 256 --n-min 2 --n-max 12 --reps 32 --out /tmp/variance_nsweep.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from variance_study import exact_grad, per_sample_grads, remaining_weights, rollout  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--V", type=int, default=5)
    p.add_argument("--T", type=int, default=256)
    p.add_argument("--n-min", type=int, default=2, help="min exponent: N=2**n_min")
    p.add_argument("--n-max", type=int, default=12, help="max exponent: N=2**n_max")
    p.add_argument("--reps", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    torch.set_default_dtype(torch.float64)

    NS = [2**k for k in range(args.n_min, args.n_max + 1)]
    N_POOL = NS[-1]

    theta = torch.randn(args.V, args.V, requires_grad=True)
    teacher_logits = torch.randn(args.V, args.V) * 1.5
    teacher = F.softmax(teacher_logits, -1)
    teacher_lp = F.log_softmax(teacher_logits, -1)

    rem = lambda G, T: G / remaining_weights(T)
    variants = {
        "raw G_t": lambda G, T: G,
        "G_t - batch_mean_t": lambda G, T: G - G.mean(0, keepdim=True),
        "G_t / remaining": rem,
        "(G_t - batch_mean_t)/remaining": lambda G, T: (G - G.mean(0, keepdim=True)) / remaining_weights(T),
        "G_t / (T-1)": lambda G, T: G / (T - 1),
        "no PG term": lambda G, T: torch.zeros_like(G),
    }

    print(f"exact grad T={args.T} ...", flush=True)
    g_star = exact_grad(theta, teacher, teacher_lp, args.T)
    print(f"rollout pool N={N_POOL} ...", flush=True)
    st_all, ac_all, rw_all = rollout(theta, teacher, teacher_lp, args.T, N_POOL)

    results = {
        name: {"N": [], "snr_mean": [], "snr_std": [], "cos_mean": [], "cos_std": []}
        for name in variants
    }

    for n in NS:
        print(f"N={n} ...", flush=True)
        snr_acc = {name: [] for name in variants}
        cos_acc = {name: [] for name in variants}
        for _ in range(args.reps):
            idx = torch.randperm(N_POOL)[:n]
            st, ac, rw = st_all[idx], ac_all[idx], rw_all[idx]
            for name, fn in variants.items():
                g = per_sample_grads(theta, teacher, st, ac, rw, fn)
                mean = g.mean(0)
                if n > 1:
                    noise = (g - mean).pow(2).sum(dim=(1, 2)).mean().sqrt()
                    snr = (mean.norm() / noise).item() if noise > 0 else float("nan")
                else:
                    snr = float("nan")
                cos = F.cosine_similarity(mean.flatten(), g_star.flatten(), dim=0).item()
                snr_acc[name].append(snr)
                cos_acc[name].append(cos)

        for name in variants:
            snrs = torch.tensor([x for x in snr_acc[name] if not math.isnan(x)])
            coss = torch.tensor(cos_acc[name])
            results[name]["N"].append(n)
            results[name]["snr_mean"].append(round(snrs.mean().item(), 6) if len(snrs) else None)
            results[name]["snr_std"].append(round(snrs.std(unbiased=False).item(), 6) if len(snrs) else None)
            results[name]["cos_mean"].append(round(coss.mean().item(), 6))
            results[name]["cos_std"].append(round(coss.std(unbiased=False).item(), 6))

    print("\nN     variant                        SNR        cos")
    for n_i, n in enumerate(NS):
        for name in variants:
            d = results[name]
            print(f"{n:<5} {name:<30} {d['snr_mean'][n_i]:8.4f}  {d['cos_mean'][n_i]:.5f}")

    payload = {
        "T": args.T,
        "N_POOL": N_POOL,
        "N_REPS": args.reps,
        "NS": NS,
        "results": results,
    }
    if args.out:
        Path(args.out).write_text(json.dumps(payload, indent=2))
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
