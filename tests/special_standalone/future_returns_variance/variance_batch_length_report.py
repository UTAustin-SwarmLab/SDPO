#!/usr/bin/env python3
"""Batch-size × length-regime credit-weight estimator study.

Sweeps micro-batch size B = 2^k under equal and unequal lengths for:
  raw G_t, G_t/(T-1), G_t-mean(G), (G-mean)/(T-1), G(r-mean(r)), no PG.

Run:
  python tests/special_standalone/future_returns_variance/variance_batch_length_report.py \\
      --T 256 --N 8192 --n-min 2 --n-max 10 --reps 48 \\
      --out tests/special_standalone/future_returns_variance/results_batch_length.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from variance_equal_vs_unequal import (  # noqa: E402
    VARIANTS,
    exact_grad,
    make_weights,
    metrics,
    per_sample_grads,
    rollout,
)


def survival_uniform(T: int, T_min: int) -> torch.Tensor:
    n_len = T - T_min + 1
    survival = torch.zeros(T, dtype=torch.float64)
    for t in range(T):
        lo = max(T_min, t + 1)
        survival[t] = max(0, T - lo + 1) / n_len
    return survival


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--V", type=int, default=5)
    p.add_argument("--T", type=int, default=256)
    p.add_argument("--N", type=int, default=8192, help="rollout pool size")
    p.add_argument("--n-min", type=int, default=2)
    p.add_argument("--n-max", type=int, default=10)
    p.add_argument("--reps", type=int, default=48)
    p.add_argument("--T-min-frac", type=float, default=0.25)
    p.add_argument("--alpha", type=float, default=0.0,
                   help="0=forward KL, 1=reverse KL, else generalized JSD")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, required=True)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    torch.set_default_dtype(torch.float64)

    BS = [2**k for k in range(args.n_min, args.n_max + 1)]
    assert max(BS) <= args.N

    theta = torch.randn(args.V, args.V, requires_grad=True)
    teacher_logits = torch.randn(args.V, args.V) * 1.5
    teacher = F.softmax(teacher_logits, -1)
    teacher_lp = F.log_softmax(teacher_logits, -1)

    T = args.T
    T_min = max(2, int(torch.ceil(torch.tensor(args.T_min_frac * T)).item()))

    settings = {
        "equal_lengths": {
            "lengths": None,
            "survival": None,
        },
        "unequal_lengths": {
            "lengths": torch.randint(T_min, T + 1, (args.N,)),
            "survival": survival_uniform(T, T_min),
        },
    }

    results = {
        "meta": {
            "alpha": args.alpha,
            "V": args.V, "T": T, "N": args.N, "BS": BS, "reps": args.reps,
            "T_min": T_min, "T_min_frac": args.T_min_frac, "seed": args.seed,
            "variants": VARIANTS,
        },
        "settings": {},
    }

    for setting, cfg in settings.items():
        print(f"\n===== {setting}  alpha={args.alpha} =====", flush=True)
        g_star = exact_grad(theta, teacher, teacher_lp, T, survival=cfg["survival"], alpha=args.alpha)
        st, ac, rw, mask = rollout(
            theta, teacher, teacher_lp, T, args.N, lengths=cfg["lengths"], alpha=args.alpha
        )
        print(f"  ||exact||={g_star.norm():.3f}  mean_len={mask.sum(1).mean():.1f}", flush=True)

        setting_out = {
            "exact_norm": g_star.norm().item(),
            "mean_len": mask.sum(1).mean().item(),
            "by_B": {},
        }

        # Full-pool reference (baseline over all N)
        full = {}
        for name in VARIANTS:
            W = make_weights(name, rw, mask, T)
            g = per_sample_grads(theta, teacher, st, ac, rw, mask, W, teacher_lp=teacher_lp, alpha=args.alpha)
            full[name] = metrics(g, g_star)
        setting_out["full_pool"] = full
        print("  full pool:", {k: round(v["cos"], 5) for k, v in full.items()}, flush=True)

        for B in BS:
            print(f"  B={B} ...", flush=True)
            acc = {name: {"cos": [], "snr": [], "mean_norm": [], "noise": []} for name in VARIANTS}
            for _ in range(args.reps):
                idx = torch.randperm(args.N)[:B]
                st_b, ac_b, rw_b, mask_b = st[idx], ac[idx], rw[idx], mask[idx]
                for name in VARIANTS:
                    W = make_weights(name, rw_b, mask_b, T)
                    g = per_sample_grads(
                        theta, teacher, st_b, ac_b, rw_b, mask_b, W, teacher_lp=teacher_lp, alpha=args.alpha
                    )
                    m = metrics(g, g_star)
                    for k in acc[name]:
                        acc[name][k].append(m[k] if k != "noise" else m["noise"])
            by_var = {}
            for name in VARIANTS:
                cos = torch.tensor(acc[name]["cos"])
                snr = torch.tensor(acc[name]["snr"])
                by_var[name] = {
                    "cos_mean": cos.mean().item(),
                    "cos_std": cos.std(unbiased=False).item(),
                    "snr_mean": snr.mean().item(),
                    "snr_std": snr.std(unbiased=False).item(),
                    "mean_norm": torch.tensor(acc[name]["mean_norm"]).mean().item(),
                    "noise": torch.tensor(acc[name]["noise"]).mean().item(),
                }
            setting_out["by_B"][str(B)] = by_var
            # quick line
            gmean = by_var["G_t - mean_t(G_t)"]["cos_mean"]
            rmean = by_var["G(r_t - mean_t(r_t))"]["cos_mean"]
            rmean_t = by_var["G(r_t - mean_t(r_t))/(T-1)"]["cos_mean"]
            rmean_rem = by_var["G(r_t - mean_t(r_t))/remaining"]["cos_mean"]
            raw = by_var["raw G_t"]["cos_mean"]
            print(
                f"    cos raw={raw:.3f}  G-meanG={gmean:.3f}  "
                f"G(r-meanr)={rmean:.3f}  /(T-1)={rmean_t:.3f}  /rem={rmean_rem:.3f}",
                flush=True,
            )

        results["settings"][setting] = setting_out

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nWrote {args.out}", flush=True)

    # Print markdown-friendly tables
    print("\n# Cosine vs batch size\n")
    for setting in settings:
        print(f"## {setting}\n")
        header = f"| B | " + " | ".join(VARIANTS) + " |"
        sep = "|" + "|".join(["---"] * (len(VARIANTS) + 1)) + "|"
        print(header)
        print(sep)
        for B in BS:
            cells = [str(B)]
            for name in VARIANTS:
                cells.append(f"{results['settings'][setting]['by_B'][str(B)][name]['cos_mean']:.4f}")
            print("| " + " | ".join(cells) + " |")
        print()


if __name__ == "__main__":
    main()
