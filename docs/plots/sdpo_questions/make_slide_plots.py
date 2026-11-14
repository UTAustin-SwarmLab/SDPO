"""One focused, large-font figure per hypothesis for the results deck (steps <= 400).

Usage:
    python docs/plots/sdpo_questions/make_slide_plots.py --cache <dir>

Reuses the cached W&B histories of make_plots.py. Prints, for every panel, each series' mean val acc over the
steps all series in the panel share, and its best single checkpoint (4-domain average acc/mean@16).
"""

import argparse
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import make_plots as M  # noqa: E402

BG = "#F7F6F2"  # deck background, so figures sit flush on the slides
INK, INK2, GRID = "#1B2330", "#4A5463", "#E2E0D8"
C = {
    "base": "#6250d6", "grpo": "#e34948", "distil": "#008300", "rover": "#e87ba4",
    "s1": "#2a78d6", "s2": "#eb6834", "s3": "#1baf7a", "s4": "#eda100",
    "alt": "#8f8e88",  # an alternative that is not carried forward
    "ink": "#1B2330",  # the final PG recipe
}
MARK = {"base": ".", "grpo": "x", "distil": "+", "rover": "v", "s1": "o", "s2": "s", "s3": "^", "s4": "D", "alt": "d", "ink": "h"}
YLABEL = {
    "val": "Val accuracy (mean@16, 4-domain avg)",
    "critic/score/mean": "Train reward (EMA)",
    "actor/grad_norm": "Grad norm (EMA, log)",
    "sdql/target_q_std": "Target-Q std (EMA, log)",
    "timing_s/step": "Seconds per step (EMA)",
    "self_distillation/future_kl_token_loss": "PG-term magnitude (EMA)",
}
LOG = {"actor/grad_norm", "sdql/target_q_std"}

# Shorthand chains used across slides.
BASE = {0.0: ["8ovi1uq2"], 0.5: ["54eer5m8"], 1.0: ["1jqglo3j"]}  # SDPO base, no env feedback, EMA 0.05
BASE_FB = {0.0: ["kvd9hund"], 0.5: ["mq08zmcp"], 1.0: ["oapzokb2", "6i11khkm"]}  # with env feedback
NAIVE_FB = {0.0: ["k2xgjxg6", "lufg2c2e", "gvdsc9b9"], 0.5: ["6s046pj5", "w3j4vvg8"], 1.0: ["g0mwg3bm", "bhpxjjvu"]}
LOO_FB = {0.0: ["1ruezpqc"], 0.5: ["sd5ltz6h"], 1.0: ["bd8gkfjm"]}
SUM_G01 = {0.0: ["0o6f01pm"], 0.5: ["nubdqvie"], 1.0: ["wormvd42"]}
SUM_G05 = {0.0: ["d5cozgd7"], 0.5: ["wn00dza0"], 1.0: ["odckkgtf"]}
MEAN_G01 = {0.0: ["pydodnbi"], 0.5: ["ckpb6svm"], 1.0: ["7rd3jzcz"]}
MEAN_G05 = {0.0: ["0q2rz5mf"], 0.5: ["7dpnqgxy"], 1.0: ["mnpamrjs"]}
MEAN_B01 = {0.0: ["34tib0zu"], 0.5: ["jntmxkfg"], 1.0: ["dce2eanv"]}
MEAN_B05 = {0.0: ["zku4zbt2"], 0.5: ["26z83fva"], 1.0: ["u6e8sst7"]}
# H4's recipe (batch leave-one-out baseline on G, summed, EMA 0.05) without env feedback (Aug 18, CISPO 0.3).
LOO_NOFB = {0.0: ["jpkpyc2i"], 0.5: ["guf2nmqk"], 1.0: ["2qs97mrb"]}
GRPO, DISTIL, ROVER = ["ifv06lhj"], ["tppb3ufq"], ["yoccbm7n"]
AN = {0.0: "α=0 · forward KL", 0.5: "α=0.5 · JSD", 1.0: "α=1 · reverse KL"}


def series(label, chain, color, ls="-"):
    return dict(label=label, chain=chain, color=color, ls=ls)


def draw(store, ax, metric, ss):
    for s in ss:
        x, y = store.series(s["chain"], metric)
        if len(x) == 0:
            continue
        c = C[s["color"]]
        if metric == "val":
            ax.plot(x, y, color=c, ls=s["ls"], lw=2.6, marker=MARK[s["color"]], ms=6, mec=BG, mew=1, label=s["label"])
        else:
            ys = 10 ** M.ema(np.log10(np.clip(y, 1e-12, None))) if metric in LOG else M.ema(y)
            ax.plot(x, ys, color=c, ls=s["ls"], lw=2.6, label=s["label"])
    if metric in LOG:
        ax.set_yscale("log")
    ax.set_facecolor(BG)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=15, length=0)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_xlim(0, M.MAX_STEP)
    ax.set_xlabel("training step", color=INK2, fontsize=15)


def report(store, name, title, ss):
    """Mean val over the shared window and best single checkpoint, per series."""
    vals = [(s, store.series(s["chain"], "val")) for s in ss]
    vals = [(s, xy) for s, xy in vals if len(xy[0])]
    if not vals:
        return
    end = min(xy[0][-1] for _, xy in vals)
    print(f"  [{title}] shared window: steps <= {end}")
    for s, (x, y) in vals:
        k = int(np.argmax(y))
        print(f"     {s['label'][:46]:46s} mean {np.mean(y[x <= end]):.3f} | best ckpt {y[k]:.3f} @{x[k]} (run ends {x[-1]})")


def fig(store, out, name, panels, height=5.0):
    """panels: list of (title, metric, [series...]). One row; shared legend on top."""
    n = len(panels)
    f, axes = plt.subplots(1, n, figsize=(16.64, height), squeeze=False, facecolor=BG)
    print(name)
    prev = None
    for ax, (title, metric, ss) in zip(axes[0], panels):
        draw(store, ax, metric, ss)
        ax.set_title(title, color=INK, fontsize=17, loc="left", pad=10)
        if metric != prev:
            ax.set_ylabel(YLABEL[metric], color=INK2, fontsize=14)
        prev = metric
        if metric == "val":
            report(store, name, title, ss)
    handles, labels = [], []
    for ax in axes[0]:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    f.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.005, 1.0), ncol=min(len(labels), 4),
             frameon=False, fontsize=15, labelcolor=INK, handlelength=2.6)
    rows = math.ceil(len(labels) / 4)
    f.tight_layout(rect=(0, 0, 1, 1 - (0.12 + 0.075 * rows) * 5.0 / height))
    f.savefig(os.path.join(out, name + ".png"), dpi=200, facecolor=BG)
    plt.close(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "slides"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    st = M.Store(args.cache)
    o = args.out

    # H1: stepwise SDPO vs RL from the environment reward.
    h1 = [series("SDPO base (stepwise, α=0.5)", BASE[0.5], "base"), series("GRPO", GRPO, "grpo"),
          series("ROVER (mbs16)", ROVER, "rover")]
    fig(st, o, "h1_sdpo_vs_rl", [("Validation accuracy", "val", h1), ("Training reward", "critic/score/mean", h1)])

    # H2: divergence direction for the stepwise loss.
    h2 = [series("α=0 · forward KL", BASE[0.0], "s1"), series("α=0.5 · JSD", BASE[0.5], "s2"),
          series("α=1 · reverse KL", BASE[1.0], "s3")]
    h2fb = [series("α=0 · forward KL", BASE_FB[0.0], "s1"), series("α=0.5 · JSD", BASE_FB[0.5], "s2"),
            series("α=1 · reverse KL", BASE_FB[1.0], "s3")]
    fig(st, o, "h2_divergence", [("No env feedback", "val", h2), ("With env feedback in the teacher", "val", h2fb)])

    # Part 2 is a ladder: each slide keeps SDPO base and the previous slide's winner, and changes one thing.
    # Colours follow the recipe: orange naive PG, yellow + batch baseline (summed), aqua + averaging (batch
    # baseline), blue group baseline (averaged, EMA 0.05), ink = final recipe (EMA 0.01); grey dashed = an
    # alternative that is not carried forward.
    # H3: naive PG term (summed returns, no baseline), env-feedback sweep.
    h3 = lambda a: [series("SDPO base (stepwise)", BASE_FB[a], "base"), series("+ PG term, naive (sum, no baseline)", NAIVE_FB[a], "s2")]  # noqa: E731
    fig(st, o, "h3_naive_pg", [(AN[0.0], "val", h3(0.0)), (AN[0.5], "val", h3(0.5)), (AN[0.0] + " · grad norm", "actor/grad_norm", h3(0.0))])

    # H4: add a batch leave-one-out baseline on G (same env-feedback sweep).
    h4 = lambda a: h3(a) + [series("+ baseline (new)", LOO_FB[a], "s4")]  # noqa: E731
    fig(st, o, "h4_baseline", [(AN[a], "val", h4(a)) for a in [0.0, 0.5, 1.0]])

    # From H5 on: no env feedback. H4's exact recipe without feedback, then the same batch baseline with averaging.
    h5 = lambda a: [series("SDPO base (stepwise)", BASE[a], "base"),  # noqa: E731
                    series("H4 recipe, with env feedback (as in H4)", LOO_FB[a], "s4", ":"),
                    series("H4 recipe, no env feedback", LOO_NOFB[a], "s4"),
                    series("+ averaged returns (new)", MEAN_B05[a], "s3")]
    fig(st, o, "h5_sum_vs_mean", [(AN[a], "val", h5(a)) for a in [0.0, 0.5, 1.0]])

    # H6: keep H5's winner (batch baseline, averaged); try the per-prompt group baseline.
    h6 = lambda a: [series("SDPO base (stepwise)", BASE[a], "base"), series("Batch baseline, averaged (from H5)", MEAN_B05[a], "s3"),  # noqa: E731
                    series("Group baseline instead (new)", MEAN_G05[a], "s1")]
    fig(st, o, "h6_group_vs_batch", [(AN[a], "val", h6(a)) for a in [0.0, 0.5, 1.0]])

    # H7: keep the group baseline; slow the teacher from EMA 0.05 to 0.01.
    h7 = lambda a: [series("SDPO base (stepwise)", BASE[a], "base"), series("Group baseline, EMA 0.05 (from H6)", MEAN_G05[a], "s1"),  # noqa: E731
                    series("+ slower teacher, EMA 0.01 (new)", MEAN_G01[a], "ink")]
    fig(st, o, "h7_teacher_rate_pg", [(AN[a], "val", h7(a)) for a in [0.0, 0.5, 1.0]])

    # H8: final PG recipe vs stepwise SDPO; the batch baseline at EMA 0.01 as the alternative.
    h8 = lambda a: [series("SDPO base (stepwise, EMA 0.05)", BASE[a], "base"),  # noqa: E731
                    series("Final PG recipe (from H7)", MEAN_G01[a], "ink"),
                    series("Batch-baseline alternative, EMA 0.01", MEAN_B01[a], "alt", "--")]
    fig(st, o, "h8_pg_vs_stepwise", [(AN[a], "val", h8(a)) for a in [0.0, 0.5, 1.0]])

    # H9: is the baseline needed once returns are averaged? Distil = forward CE + averaged PG, no baseline.
    h9 = [series("SDPO base (stepwise, α=0)", BASE[0.0], "base"),
          series("Final PG recipe (α=0, from H8)", MEAN_G01[0.0], "ink"),
          series("Distil: forward CE + averaged PG, no baseline", DISTIL, "distil")]
    fig(st, o, "h9_distil", [("Validation accuracy", "val", h9), ("Training reward", "critic/score/mean", h9)])

    # H10: SDQL out of the box (no centring, EMA 0.05) vs SDPO references at EMA 0.05.
    h10 = {0.0: ["nh85fcx4"], 0.5: ["33b8ef1m"], 1.0: ["s0zyvxdm"]}
    h10o = {0.0: ["x1yt14um"], 0.5: ["myb1g10y"], 1.0: ["k41rpp2s"]}
    h10s = lambda a: [series("SDPO base (stepwise)", BASE[a], "base"), series("SDPO + PG, group, averaged, EMA 0.05", MEAN_G05[a], "s1"),  # noqa: E731
                      series("SDQL · uniform target", h10[a], "s3"), series("SDQL · on-policy target", h10o[a], "s2")]
    fig(st, o, "h10_sdql_out_of_box", [(AN[a], "val", h10s(a)) for a in [0.0, 0.5, 1.0]])

    # H11: reward centring, matched pairs.
    pairs = [("Uniform target · α=0.5", ["t8qss8xb"], ["xwmz3igm"]), ("Uniform target · α=1", ["7ztqehbg"], ["t0vtrfiu"]),
             ("On-policy target · α=0.5", ["y8eq5q0o"], ["3a50a9ot"]), ("On-policy target · α=1", ["y9ddre11"], ["zm03pq9q"])]
    fig(st, o, "h11_centring", [(t, "val", [series("With reward centring", w, "s1"), series("Without centring", wo, "s2")])
                                 for t, w, wo in pairs])

    # H12: lambda returns (no centring, EMA 0.05).
    lam = {0.5: [["33b8ef1m"], ["myb1g10y"], ["l60ea65a"], ["14qw269p"]], 1.0: [["s0zyvxdm"], ["k41rpp2s"], ["ezh5bdi7"], ["597ib0aq"]]}
    lab = ["Uniform (1-step)", "On-policy (1-step)", "λ-return, λ=0.9", "λ-return, λ=1.0"]
    col, sty = ["s3", "s2", "s4", "alt"], ["-", "-", "-", "--"]
    h12 = lambda a: [series(lab[i], lam[a][i], col[i], sty[i]) for i in range(4)]  # noqa: E731
    fig(st, o, "h12_lambda", [(AN[0.5], "val", h12(0.5)), (AN[1.0], "val", h12(1.0)), (AN[0.5] + " · target-Q spread", "sdql/target_q_std", h12(0.5))])

    # H13: teacher rate for SDQL (on-policy + centring, fx2).
    tr = {0.5: [["12ocxwrk"], ["7jm6nqe7"], ["1m6vtuxy"]], 1.0: [["pquoaf1g"], ["xzn0voxu"], ["klxawodx"]]}
    h13 = lambda a: [series("Teacher frozen (EMA 0)", tr[a][0], "s3"), series("Teacher EMA 0.01", tr[a][1], "s1"),  # noqa: E731
                     series("Teacher EMA 0.05", tr[a][2], "s2")]
    fig(st, o, "h13_teacher_rate_sdql", [(AN[0.5], "val", h13(0.5)), (AN[1.0], "val", h13(1.0))])

    # H14: mini-batch size for SDQL (on-policy + centring, EMA 0.01).
    mb = {0.5: [["7jm6nqe7"], ["ooqw4j1s"], ["9ntvz0tx"]], 1.0: [["xzn0voxu"], ["wbl6mkjj"], ["uj948anc"]]}
    h14 = lambda a: [series("Mini-batch 32 (1 update / step)", mb[a][0], "s2"), series("Mini-batch 16 (2 updates)", mb[a][1], "s1"),  # noqa: E731
                     series("Mini-batch 8 (4 updates)", mb[a][2], "s3")]
    fig(st, o, "h14_minibatch_sdql", [(AN[0.5], "val", h14(0.5)), (AN[1.0], "val", h14(1.0)), ("Cost per step (α=0.5)", "timing_s/step", h14(0.5))])


if __name__ == "__main__":
    main()
