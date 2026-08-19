#!/usr/bin/env python3
"""Compare credit-weight rules for the future-return PG term (Markov toy).

Policy depends only on the previous token (V states). Exact grad of

    J = sum_t sum_s P_t(s) D_alpha(pi(.|s), teacher(.|s))

is available in closed form, with D_alpha matching core_algos full-logit
``compute_self_distillation_loss``:

    alpha=0   forward KL  KL(teacher || pi)
    alpha=1   reverse KL  KL(pi || teacher)
    else      generalized JSD  (1-a) KL(pi || m) + a KL(teacher || m)
              m = (1-a) pi + a teacher   (default a=0.5)

Per-sample grads are autograd of the on-policy training surrogate
(per-token D_alpha + W_t log pi of the student-sampled token).

Run from repo root:
    python tests/special_standalone/future_returns_variance/variance_study.py
    python tests/special_standalone/future_returns_variance/variance_study.py --alpha 1
    python tests/special_standalone/future_returns_variance/variance_study.py --alpha 0.5 --T 64 --N 2000
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F


def remaining_weights(T: int) -> torch.Tensor:
    t = torch.arange(T)
    return (T - t - 1).clamp(min=1).to(torch.float64)


def remaining_from_mask(mask: torch.Tensor) -> torch.Tensor:
    """DISTIL remaining: (L_i - t - 1).clamp(min=1), L_i = mask.sum(1)."""
    T = mask.shape[1]
    positions = torch.arange(T, device=mask.device, dtype=mask.dtype).unsqueeze(0)
    total_len = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
    return (total_len - positions - 1).clamp(min=1.0)


def future_returns(rewards: torch.Tensor) -> torch.Tensor:
    """G_t = sum_{k>t} r_k  (gamma=1, shifted; matches core_algos)."""
    shifted = torch.cat([rewards[:, 1:], torch.zeros_like(rewards[:, :1])], 1)
    return torch.flip(torch.cumsum(torch.flip(shifted, [1]), 1), [1])


def divergence_name(alpha: float) -> str:
    if alpha == 0.0:
        return "forward_kl"
    if alpha == 1.0:
        return "reverse_kl"
    return f"jsd_alpha_{alpha:g}"


def token_divergence(
    log_pi: torch.Tensor,
    teacher: torch.Tensor,
    teacher_lp: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Per-state D_alpha matching full-logit ``compute_self_distillation_loss``.

    ``log_pi``, ``teacher``, ``teacher_lp`` broadcast on the last (vocab) dim.
    """
    if alpha == 0.0:
        return (teacher * (teacher_lp - log_pi)).sum(-1)
    if alpha == 1.0:
        return (log_pi.exp() * (log_pi - teacher_lp)).sum(-1)
    a = log_pi.new_tensor(alpha)
    mix_lp = torch.logsumexp(
        torch.stack([log_pi + (1.0 - a).log(), teacher_lp + a.log()]), dim=0
    )
    kl_teacher = (teacher * (teacher_lp - mix_lp)).sum(-1)
    kl_student = (log_pi.exp() * (log_pi - mix_lp)).sum(-1)
    return torch.lerp(kl_student, kl_teacher, a)


def exact_grad(
    theta: torch.Tensor,
    teacher: torch.Tensor,
    teacher_lp: torch.Tensor,
    T: int,
    alpha: float = 0.0,
    survival: torch.Tensor | None = None,
) -> torch.Tensor:
    log_pi = F.log_softmax(theta, -1)
    pi = log_pi.exp()
    d = token_divergence(log_pi, teacher, teacher_lp, alpha)
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
def rollout(theta, teacher, teacher_lp, T: int, N: int, alpha: float = 0.0,
            lengths: torch.Tensor | None = None):
    """On-policy rollouts: a_t ~ pi(.|s_t), s_{t+1} = a_t.

    Rewards are per-token D_alpha. Optional ``lengths`` right-pads to T.
    """
    V = theta.shape[0]
    log_pi = F.log_softmax(theta, -1)
    pi = log_pi.exp()
    d = token_divergence(log_pi, teacher, teacher_lp, alpha)
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
        return states, actions, rewards
    t = torch.arange(T).unsqueeze(0)
    mask = (t < lengths.unsqueeze(1)).to(rewards.dtype)
    return states, actions, rewards * mask, mask


def surrogate_grads(
    theta,
    teacher,
    states,
    actions,
    W,
    mask=None,
    teacher_lp=None,
    alpha: float = 0.0,
    chunk: int = 512,
):
    """Autograd of the on-policy SDPO full-logit surrogate. Returns [N, V, V].

        L_i = sum_t mask_t * [ D_alpha(s_t) + W_t * log pi(a_t|s_t) ]

    Matches ``compute_self_distillation_loss`` with full_logit_distillation
    and use_future_returns=True. W is stop-grad.
    """
    V = theta.shape[0]
    N = states.shape[0]
    if teacher_lp is None:
        teacher_lp = teacher.clamp_min(1e-300).log()
    theta_d = theta.detach()
    out = torch.empty(N, V, V, dtype=theta_d.dtype)

    for lo in range(0, N, chunk):
        hi = min(lo + chunk, N)
        st, ac, w = states[lo:hi], actions[lo:hi], W[lo:hi].detach()
        theta_b = theta_d.unsqueeze(0).expand(hi - lo, V, V).clone().requires_grad_(True)
        log_pi = F.log_softmax(theta_b, dim=-1)
        log_pi_s = log_pi.gather(1, st.unsqueeze(-1).expand(-1, -1, V))
        div = token_divergence(log_pi_s, teacher[st], teacher_lp[st], alpha)
        log_pi_a = log_pi_s.gather(-1, ac.unsqueeze(-1)).squeeze(-1)
        per_token = div + w * log_pi_a
        if mask is not None:
            per_token = per_token * mask[lo:hi]
        out[lo:hi] = torch.autograd.grad(per_token.sum(), theta_b)[0]
    return out


def per_sample_grads(
    theta, teacher, states, actions, rewards, weight_fn,
    teacher_lp=None, alpha: float = 0.0, chunk: int = 512,
):
    """Per-trajectory grads [N,V,V] for ``weight_fn(G, T, rewards)``."""
    T = rewards.shape[1]
    G = future_returns(rewards)
    W = weight_fn(G, T, rewards)
    return surrogate_grads(
        theta, teacher, states, actions, W,
        teacher_lp=teacher_lp, alpha=alpha, chunk=chunk,
    )


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
    p.add_argument("--alpha", type=float, default=0.0,
                   help="0=forward KL, 1=reverse KL, else generalized JSD (0.5=JS)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    torch.set_default_dtype(torch.float64)

    theta = torch.randn(args.V, args.V, requires_grad=True)
    teacher_logits = torch.randn(args.V, args.V) * 1.5
    teacher = F.softmax(teacher_logits, -1)
    teacher_lp = F.log_softmax(teacher_logits, -1)

    rem = lambda G, T, r: G / remaining_weights(T)
    center_r = lambda G, T, r: future_returns(r - r.mean(0, keepdim=True))
    variants = {
        "raw G_t": lambda G, T, r: G,
        "G_t - batch_mean_t": lambda G, T, r: G - G.mean(0, keepdim=True),
        "G(r - batch_mean_t)": center_r,
        "G_t / remaining": rem,
        "(G_t - batch_mean_t)/remaining": lambda G, T, r: (G - G.mean(0, keepdim=True)) / remaining_weights(T),
        "G(r - batch_mean_t)/remaining": lambda G, T, r: center_r(G, T, r) / remaining_weights(T),
        "G_t / (T-1)": lambda G, T, r: G / (T - 1),
        "G(r - batch_mean_t)/(T-1)": lambda G, T, r: center_r(G, T, r) / (T - 1),
        "no PG term": lambda G, T, r: torch.zeros_like(G),
    }

    print(f"divergence={divergence_name(args.alpha)}  alpha={args.alpha}")
    for T in args.T:
        g_star = exact_grad(theta, teacher, teacher_lp, T, alpha=args.alpha)
        st, ac, rw = rollout(theta, teacher, teacher_lp, T, args.N, alpha=args.alpha)
        print(f"\n=== {divergence_name(args.alpha)}  T={T}  N={args.N}  "
              f"||exact grad||={g_star.norm():.3f}  mean d_t={rw.mean():.4f} ===")
        for name, fn in variants.items():
            g = per_sample_grads(
                theta, teacher, st, ac, rw, fn, teacher_lp=teacher_lp, alpha=args.alpha
            )
            report(name, g, g_star)


if __name__ == "__main__":
    main()
