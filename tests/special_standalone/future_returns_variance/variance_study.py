#!/usr/bin/env python3
"""Compare credit-weight rules for the future-return PG term (Markov KL toy).

Policy depends only on the previous token (V states). Exact grad of

    J = sum_t sum_s P_t(s) KL(teacher(.|s) || pi(.|s))

is available in closed form. Per-sample REINFORCE grads are analytic for
tabular softmax; we compare choices of credit weight W_t on G_t = sum_{k>t} r_k.

Run from repo root:
    python tests/special_standalone/future_returns_variance/variance_study.py
    python tests/special_standalone/future_returns_variance/variance_study.py --T 64 --N 2000
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F


def remaining_weights(T: int) -> torch.Tensor:
    t = torch.arange(T)
    return (T - t - 1).clamp(min=1).to(torch.float64)


def exact_grad(theta: torch.Tensor, teacher: torch.Tensor, teacher_lp: torch.Tensor, T: int) -> torch.Tensor:
    pi = F.softmax(theta, -1)
    d = (teacher * (teacher_lp - F.log_softmax(theta, -1))).sum(-1)
    P = torch.full((theta.shape[0],), 1.0 / theta.shape[0], dtype=theta.dtype)
    J = theta.new_zeros(())
    for _ in range(T):
        J = J + (P * d).sum()
        P = (P.unsqueeze(-1) * pi).sum(0)
        P = P / P.sum()
    return torch.autograd.grad(J, theta)[0]


@torch.no_grad()
def rollout(theta, teacher, teacher_lp, T: int, N: int):
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
    return states, actions, d[states]


@torch.no_grad()
def per_sample_grads(theta, teacher, states, actions, rewards, weight_fn):
    V = theta.shape[0]
    N, T = rewards.shape
    pi = F.softmax(theta, -1)
    shifted = torch.cat([rewards[:, 1:], torch.zeros_like(rewards[:, :1])], 1)
    G = torch.flip(torch.cumsum(torch.flip(shifted, [1]), 1), [1])
    W = weight_fn(G, T)
    onehot = F.one_hot(actions, V).to(rewards.dtype)
    pi_s = pi[states]
    contrib = (pi_s - teacher[states]) + W.unsqueeze(-1) * (onehot - pi_s)
    g = torch.zeros(N, V, V, dtype=rewards.dtype)
    g.scatter_add_(1, states.unsqueeze(-1).expand(-1, -1, V), contrib)
    return g


def report(name: str, g: torch.Tensor, g_star: torch.Tensor) -> float:
    mean = g.mean(0)
    noise = (g - mean).pow(2).sum(dim=(1, 2)).mean().sqrt()
    snr = (mean.norm() / noise).item()
    cos = F.cosine_similarity(mean.flatten(), g_star.flatten(), dim=0).item()
    print(f"  {name:<32} SNR={snr:9.4f}  cos_to_exact={cos:.5f}")
    return snr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--V", type=int, default=5)
    p.add_argument("--T", type=int, nargs="+", default=[64, 256, 1024])
    p.add_argument("--N", type=int, default=20000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    torch.set_default_dtype(torch.float64)

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

    for T in args.T:
        g_star = exact_grad(theta, teacher, teacher_lp, T)
        st, ac, rw = rollout(theta, teacher, teacher_lp, T, args.N)
        print(f"\n=== T={T}  N={args.N}  ||exact grad||={g_star.norm():.3f}  "
              f"mean d_t={rw.mean():.4f} ===")
        for name, fn in variants.items():
            g = per_sample_grads(theta, teacher, st, ac, rw, fn)
            report(name, g, g_star)


if __name__ == "__main__":
    main()
