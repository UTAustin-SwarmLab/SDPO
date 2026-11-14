"""Plots for the SDPO / SDQL research questions (bs=32, mini-batch=32 runs only, steps <= 400).

Usage:
    python docs/plots/sdpo_questions/make_plots.py --cache <dir> [--pull]

--pull downloads run metadata + histories from the two W&B projects into --cache;
without it the cached JSON is reused.
"""

import argparse
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PROJECTS = ["harshg5799/SDPO2-hrxu", "harshg5799/SDPO2-harshgoel99"]
MAX_STEP = 400
GS = "training/global_step"
DOMAINS = ["biology", "chemistry", "material", "physics"]
VAL_KEYS = [f"val-core/sciknoweval_{d}/acc/mean@16" for d in DOMAINS]
TRAIN_KEYS = ["critic/score/mean", "actor/grad_norm", "actor/entropy", "actor/pg_loss", "response_length/mean"]
OPT_KEYS = [
    "self_distillation/future_kl_token_loss",
    "self_distillation/future_returns_reward_baseline",
    "self_distillation/future_returns_baseline",
    "self_distillation/future_returns_cispo_clipfrac",
    "sdql/target_q_mean",
    "sdql/target_q_std",
    "sdql/reward_baseline",
    "sdql/reward_forward_std",
    "sdql/reward_forward_mean",
    "rollout_corr/kl",
    "timing_s/step",
]

# Reference categorical palette (dataviz skill, light mode), fixed slot order.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MARKERS = ["o", "s", "^", "D"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e6e5e0"

ALPHA_NAME = {0.0: "α=0 · forward KL", 0.5: "α=0.5 · JSD", 1.0: "α=1 · reverse KL"}

# Reference methods, drawn under the coloured series of each question with a fixed colour per method
# (palette slots 5-8, so they never collide with the question series in slots 1-4).
# mbs32 where a usable run exists; ROVER has none past step 22 at mbs32, so it falls back to mbs16.
REF_STYLE = {
    "SDPO base (stepwise only)": dict(color="#6250d6", ls="-", marker="."),
    "GRPO": dict(color="#e34948", ls="--", marker="x"),
    "Distil (α=1)": dict(color="#008300", ls="-.", marker="+"),
    "ROVER (mbs16)": dict(color="#e87ba4", ls=":", marker="v"),
}
FIXED_REFS = [("GRPO", ["ifv06lhj"]), ("Distil (α=1)", ["tppb3ufq"]), ("ROVER (mbs16)", ["yoccbm7n"])]
REF_EXTRA_IDS = ["yoccbm7n"]  # outside the mbs32 filter in pull()
# SDQL at mini-batch 16 / 8 (ofp sweep, Aug 21): identical to the mbs32 fx2 sweep except ppo_mini_batch_size.
SDQL_SMALL_MBS_IDS = ["ooqw4j1s", "wbl6mkjj", "9ntvz0tx", "uj948anc", "q9zov1au", "5mz5a4fb", "z8610td1", "39ws2bjy"]
REF_METRICS_TRAIN = {"critic/score/mean"}
REF_NOTE = ("References: GRPO ifv06lhj (mbs32, lr 3e-6, ends at step 199); Distil tppb3ufq (mbs32, α=1, lr 1e-5, teacher EMA 0.01);\n"
            "ROVER yoccbm7n (mbs16, lr 1e-5); SDPO base = stepwise-only SDPO (mbs32, lr 1e-5, teacher EMA 0.05).")


# --------------------------------------------------------------------------- data


def pull(cache):
    import wandb

    api = wandb.Api(timeout=120)
    meta = []
    for p in PROJECTS:
        for r in api.runs(p, per_page=500):
            meta.append(
                dict(project=p, id=r.id, name=r.name, state=r.state, created=r.created_at, config=r.config,
                     summary={k: v for k, v in r.summary._json_dict.items() if not isinstance(v, dict)})
            )
    os.makedirs(os.path.join(cache, "hist"), exist_ok=True)
    json.dump(meta, open(os.path.join(cache, "meta.json"), "w"), default=str)

    sel = [r for r in meta if r["config"]
           and ((r["config"]["actor_rollout_ref"]["actor"].get("ppo_mini_batch_size") == 32
                 and r["config"]["data"].get("train_batch_size") == 32) or r["id"] in REF_EXTRA_IDS + SDQL_SMALL_MBS_IDS)]

    def scan(run, keys):
        return [{k: row.get(k) for k in keys} for row in run.scan_history(keys=keys, page_size=2000)]

    def work(r):
        run = api.run(f"{r['project']}/{r['id']}")
        out = {"train": scan(run, [GS] + TRAIN_KEYS), "val": scan(run, [GS] + VAL_KEYS)}
        for k in OPT_KEYS:
            if k in r["summary"]:
                out[k] = scan(run, [GS, k])
        json.dump(out, open(os.path.join(cache, "hist", r["id"] + ".json"), "w"))

    with ThreadPoolExecutor(12) as ex:
        list(ex.map(work, sel))


def num(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return math.nan
    return v if math.isfinite(v) else math.nan


class Store:
    def __init__(self, cache):
        self.cache = cache
        self.meta = {r["id"]: r for r in json.load(open(os.path.join(cache, "meta.json")))}
        self._h = {}

    def hist(self, rid):
        if rid not in self._h:
            self._h[rid] = json.load(open(os.path.join(self.cache, "hist", rid + ".json")))
        return self._h[rid]

    def series(self, chain, metric):
        """Stitch a chain of run ids (resumes) into one {step: value}; later runs win on overlap."""
        out = {}
        # "val" = 4-domain average of acc/mean@16; "val/<domain>" = that domain's acc/mean@16 only.
        keys = [VAL_KEYS[DOMAINS.index(metric.split("/", 1)[1])]] if metric.startswith("val/") else VAL_KEYS
        for rid in chain:
            h = self.hist(rid)
            if metric.startswith("val"):
                for row in h["val"]:
                    vals = [num(row.get(k)) for k in keys]
                    s = row.get(GS)
                    if s is not None and all(not math.isnan(v) for v in vals):
                        out[int(s)] = float(np.mean(vals))
            else:
                rows = h["train"] if metric in TRAIN_KEYS else h.get(metric, [])
                if isinstance(rows, dict):
                    rows = []
                for row in rows:
                    s, v = row.get(GS), num(row.get(metric))
                    if s is not None and not math.isnan(v):
                        out[int(s)] = v
        steps = np.array(sorted(s for s in out if s <= MAX_STEP))
        return steps, np.array([out[s] for s in steps])


def ema(y, a=0.9):
    out, m = np.empty_like(y), None
    for i, v in enumerate(y):
        m = v if m is None else a * m + (1 - a) * v
        out[i] = m
    return out


# --------------------------------------------------------------------------- plotting


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8, length=0)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xlim(0, MAX_STEP)


METRIC_LABEL = {
    "val": "Val acc (mean@16, 4-domain avg)",
    "critic/score/mean": "Train score (EMA)",
    "actor/grad_norm": "Grad norm (EMA, log)",
    "actor/entropy": "Entropy (EMA)",
    "self_distillation/future_kl_token_loss": "Future-return PG term (EMA)",
    "sdql/target_q_std": "Target-Q std (EMA, log)",
    "response_length/mean": "Response length (EMA)",
    "timing_s/step": "Wall-clock s/step (EMA)",
    **{f"val/{d}": f"{d.capitalize()}\nacc/mean@16" for d in DOMAINS},
}
DOMAIN_METRICS = [f"val/{d}" for d in DOMAINS]
LOG_METRICS = {"actor/grad_norm", "sdql/target_q_std"}


def plot_panel(ax, store, series_specs, metric, refs=()):
    """series_specs: list of (label, chain); refs: list of (label, chain) drawn as grey context lines."""
    if metric.startswith("val") or metric in REF_METRICS_TRAIN:
        for label, chain in refs:
            x, y = store.series(chain, metric)
            if len(x) == 0:
                continue
            st = REF_STYLE[label]
            if metric.startswith("val"):
                ax.plot(x, y, color=st["color"], ls=st["ls"], lw=1.4, marker=st["marker"], ms=4, label=label, zorder=1)
            else:
                ax.plot(x, ema(y), color=st["color"], ls=st["ls"], lw=1.4, label=label, zorder=1)
    for i, (label, chain) in enumerate(series_specs):
        if not chain:
            continue
        x, y = store.series(chain, metric)
        if len(x) == 0:
            continue
        c = SERIES[i]
        if metric.startswith("val"):
            ax.plot(x, y, color=c, lw=2, marker=MARKERS[i], ms=4.5, mec=SURFACE, mew=1, label=label)
        else:
            if metric in LOG_METRICS:
                y = np.log10(np.clip(y, 1e-12, None))
                ys = 10 ** ema(y)
            else:
                ys = ema(y)
            ax.plot(x, ys, color=c, lw=2, label=label)
    if metric in LOG_METRICS:
        ax.set_yscale("log")
    style_axes(ax)


def val_summary(store, chain, end=None, metric="val"):
    x, y = store.series(chain, metric)
    if len(x) == 0:
        return None
    if end is not None:
        m = x <= end
        x, y = x[m], y[m]
    if len(x) == 0:
        return None
    return dict(last_step=int(x[-1]), final=float(np.mean(y[-3:])), mean=float(np.mean(y)), best=float(np.max(y)))


def references(alphas, sdpo_base=None):
    """Per-column reference lines. sdpo_base: {alpha: chain} when SDPO base is not already a coloured series."""
    return [([("SDPO base (stepwise only)", sdpo_base[a])] if sdpo_base else []) + FIXED_REFS for a in alphas]


def note_layout(fig, notes):
    """Bottom margin (figure fraction) for footnote lines, sized in inches so it holds at any figure height."""
    if not notes:
        return 0
    fig.text(0.01, 0.005, notes, fontsize=7.5, color=INK2, va="bottom")
    return (0.2 + 0.16 * notes.count("\n")) / fig.get_figheight()


def figure_grid(store, title, subtitle, columns, series_labels, metrics, fname, outdir, notes=None, by_domain=True, refs=None):
    """columns: list of (column title, [chain per series label]); refs: per-column reference lists (see references()).

    With by_domain, also writes <fname>_by_domain.png: one row per domain's acc/mean@16, same runs and columns.
    """
    summary = []
    if by_domain:
        summary += figure_grid(store, title + " · per domain", subtitle + " Rows: each domain's own acc/mean@16.",
                               columns, series_labels, DOMAIN_METRICS, fname.replace(".png", "_by_domain.png"),
                               outdir, notes=notes, by_domain=False, refs=refs)
    refs = refs or [[] for _ in columns]
    if any(refs):
        notes = f"{notes}\n{REF_NOTE}" if notes else REF_NOTE
    nr, nc = len(metrics), len(columns)
    fig, axes = plt.subplots(nr, nc, figsize=(4.3 * nc, 2.7 * nr + 1.3), squeeze=False, facecolor=SURFACE)
    for j, (ctitle, chains) in enumerate(columns):
        specs = list(zip(series_labels, chains))
        for i, metric in enumerate(metrics):
            ax = axes[i, j]
            plot_panel(ax, store, specs, metric, refs[j])
            if i == 0:
                ax.set_title(ctitle, color=INK, fontsize=10, loc="left")
            if j == 0:
                ax.set_ylabel(METRIC_LABEL.get(metric, metric), color=INK2, fontsize=8.5)
            if i == nr - 1:
                ax.set_xlabel("training step", color=INK2, fontsize=8.5)
        # common window = shortest series in this column, so numbers compare like-for-like
        ends = [val_summary(store, ch)["last_step"] for ch in chains if ch and val_summary(store, ch)]
        common = min(ends) if ends else None
        for vm in [m for m in metrics if m.startswith("val")]:
            for lab, ch in specs + refs[j]:
                if not ch:
                    continue
                full, win = val_summary(store, ch, metric=vm), val_summary(store, ch, common, metric=vm)
                if full:
                    summary.append(dict(figure=fname, metric=vm, column=ctitle, series=lab, runs=",".join(ch),
                                        common_end=common, **{f"win_{k}": v for k, v in (win or {}).items()},
                                        **{f"full_{k}": v for k, v in full.items()}))
    handles, labels = [], []
    for ax in axes.flat:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    order = [labels.index(s) for s in list(series_labels) + list(REF_STYLE) if s in labels]
    ncol = min(4 if nc > 2 else 2, len(order))
    legend_rows = -(-len(order) // ncol)
    fig.legend([handles[k] for k in order], [labels[k] for k in order], loc="upper left",
               bbox_to_anchor=(0.01, 1 - 0.55 / fig.get_figheight()), ncol=ncol, frameon=False,
               fontsize=9, labelcolor=INK)
    fig.text(0.01, 1 - 0.12 / fig.get_figheight(), title, fontsize=13, color=INK, weight="bold", va="top")
    fig.text(0.01, 1 - 0.38 / fig.get_figheight(), subtitle, fontsize=9, color=INK2, va="top")
    bottom = note_layout(fig, notes)
    fig.tight_layout(rect=(0, bottom, 1, 1 - (0.7 + 0.22 * legend_rows) / fig.get_figheight()))
    fig.savefig(os.path.join(outdir, fname), dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return summary


# --------------------------------------------------------------------------- best val per method


def method_of(store, rid):
    """Method family from the run config (Qwen3-8B only; None = not part of the table)."""
    r = store.meta[rid]
    c = r["config"]
    if "Olmo" in c["actor_rollout_ref"]["model"]["path"] or not os.path.exists(os.path.join(store.cache, "hist", rid + ".json")):
        return None
    a = c["actor_rollout_ref"]["actor"]
    sd = a.get("self_distillation") or {}
    mode = a["policy_loss"]["loss_mode"]
    if rid in SDQL_SMALL_MBS_IDS:
        return "SDQL (mbs 8/16)"
    if a.get("ppo_mini_batch_size") != 32 and rid not in REF_EXTRA_IDS:
        return None
    if mode == "vanilla":
        return "GRPO"
    if mode == "distil":
        return "Distil"
    if mode == "rover":
        return "ROVER"
    if mode == "sdql":
        return "SDQL (mbs 32)"
    if mode == "sdpo" and sd.get("full_logit_distillation", True):
        return "SDPO + future returns" if (sd.get("use_future_returns") or sd.get("use_gae")) else "SDPO base (stepwise only)"
    return None


def best_tables(store, out):
    """Per method, one checkpoint: the (run, step <= MAX_STEP) with the highest 4-domain average val acc/mean@16,
    reported with that same checkpoint's per-domain values. Never mixes runs or steps across domains."""
    methods = ["GRPO", "Distil", "ROVER", "SDPO base (stepwise only)", "SDPO + future returns", "SDQL (mbs 32)", "SDQL (mbs 8/16)"]
    best = {}  # method -> (avg4, rid, step)
    for rid in store.meta:
        m = method_of(store, rid)
        if m is None:
            continue
        x, y = store.series([rid], "val")
        if len(y) and (m not in best or y.max() > best[m][0]):
            k = int(np.argmax(y))
            best[m] = (float(y[k]), rid, int(x[k]))

    def alpha(rid):
        return (store.meta[rid]["config"]["actor_rollout_ref"]["actor"].get("self_distillation") or {}).get("alpha")

    rows = []
    with open(os.path.join(out, "best_val_per_method.csv"), "w") as f:
        f.write("method," + ",".join(DOMAINS) + ",avg4,run,step,alpha,name\n")
        for m in methods:
            if m not in best:
                continue
            avg4, rid, step = best[m]
            dom = []
            for metric in DOMAIN_METRICS:
                x, y = store.series([rid], metric)
                dom.append(float(y[list(x).index(step)]))
            rows.append((m, dom, avg4, rid, step))
            f.write(f"{m}," + ",".join(f"{v:.4f}" for v in dom) + f",{avg4:.4f},{rid},{step},{alpha(rid)},{store.meta[rid]['name'][:80]}\n")
    return rows


# Every variant that was run (Qwen3-8B, train batch 32), as (group, variant, {alpha: chain}).
# Chains list resumed runs in order. mbs32 unless the group says otherwise.
VARIANTS = [
    ("References", "GRPO (lr 3e-6; run ends at step 199)", {0.5: ["ifv06lhj"]}),
    ("References", "Distil (lr 1e-5, teacher EMA 0.01)", {1.0: ["tppb3ufq"]}),
    ("References", "ROVER (mbs16, lr 1e-5)", {0.5: ["yoccbm7n"]}),
    ("SDPO base (stepwise only), lr 1e-5, EMA 0.05", "With env feedback", {0.0: ["kvd9hund"], 0.5: ["mq08zmcp"], 1.0: ["oapzokb2", "6i11khkm"]}),
    ("SDPO base (stepwise only), lr 1e-5, EMA 0.05", "No env feedback", {0.0: ["8ovi1uq2"], 0.5: ["54eer5m8"], 1.0: ["1jqglo3j"]}),
    ("SDPO + future returns, env feedback (Aug 8-11), EMA 0.05", "No baseline",
     {0.0: ["k2xgjxg6", "lufg2c2e", "gvdsc9b9"], 0.5: ["6s046pj5", "w3j4vvg8"], 1.0: ["g0mwg3bm", "bhpxjjvu"]}),
    ("SDPO + future returns, env feedback (Aug 8-11), EMA 0.05", "Baseline v1 (pre-fix)",
     {0.0: ["3e71h4jo"], 0.5: ["epl4sx9x"], 1.0: ["zqa4ysx8", "l3zcepw9"]}),
    ("SDPO + future returns, env feedback (Aug 8-11), EMA 0.05", "Batch LOO baseline on G (corrected)",
     {0.0: ["1ruezpqc"], 0.5: ["sd5ltz6h"], 1.0: ["bd8gkfjm"]}),
    ("SDPO + future returns, no feedback, centred r_t (Aug 22-30)", "Sum, group LOO, CISPO 0.2, EMA 0.05",
     {0.0: ["v1w1gv0d"], 0.5: ["moyab5n9"], 1.0: ["yegf3zsn"]}),
    ("SDPO + future returns, no feedback, centred r_t (Aug 22-30)", "Sum, group LOO, CISPO 0.2, EMA 0.01",
     {0.0: ["q1f1fs4n"], 0.5: ["amvqhgzu"], 1.0: ["lwx0hu7z"]}),
    ("SDPO + future returns, no feedback, centred r_t (Aug 22-30)", "Sum, group LOO, CISPO 0.99, EMA 0.05",
     {0.0: ["d5cozgd7"], 0.5: ["wn00dza0"], 1.0: ["odckkgtf"]}),
    ("SDPO + future returns, no feedback, centred r_t (Aug 22-30)", "Sum, group LOO, CISPO 0.99, EMA 0.01",
     {0.0: ["0o6f01pm"], 0.5: ["nubdqvie"], 1.0: ["wormvd42"]}),
    ("SDPO + future returns, no feedback, centred r_t (Aug 22-30)", "Mean (remaining), group LOO, CISPO 0.99, EMA 0.05",
     {0.0: ["0q2rz5mf"], 0.5: ["7dpnqgxy"], 1.0: ["mnpamrjs"]}),
    ("SDPO + future returns, no feedback, centred r_t (Aug 22-30)", "Mean (remaining), group LOO, CISPO 0.99, EMA 0.01",
     {0.0: ["pydodnbi"], 0.5: ["ckpb6svm"], 1.0: ["7rd3jzcz"]}),
    ("SDPO + future returns, no feedback, centred r_t (Aug 22-30)", "Mean (remaining), batch LOO, CISPO 0.99, EMA 0.05",
     {0.0: ["zku4zbt2"], 0.5: ["26z83fva"], 1.0: ["u6e8sst7"]}),
    ("SDPO + future returns, no feedback, centred r_t (Aug 22-30)", "Mean (remaining), batch LOO, CISPO 0.99, EMA 0.01",
     {0.0: ["34tib0zu"], 0.5: ["jntmxkfg"], 1.0: ["dce2eanv"]}),
    ("SDQL, lr 3e-6, EMA 0.05, Sep 5-6 sweep", "Uniform, no centring", {0.0: ["nh85fcx4"], 0.5: ["33b8ef1m"], 1.0: ["s0zyvxdm"]}),
    ("SDQL, lr 3e-6, EMA 0.05, Sep 5-6 sweep", "On-policy, no centring", {0.0: ["x1yt14um"], 0.5: ["myb1g10y"], 1.0: ["k41rpp2s"]}),
    ("SDQL, lr 3e-6, EMA 0.05, Sep 5-6 sweep", "λ=0.9, no centring", {0.0: ["yc3flj5o"], 0.5: ["l60ea65a"], 1.0: ["ezh5bdi7"]}),
    ("SDQL, lr 3e-6, EMA 0.05, Sep 5-6 sweep", "λ=1.0, no centring", {0.0: ["eqdmei36"], 0.5: ["14qw269p"], 1.0: ["597ib0aq"]}),
    ("SDQL, lr 3e-6, EMA 0.05, Sep 5-6 sweep", "Uniform, centring", {0.0: ["mgr7x5ab"]}),
    ("SDQL, lr 3e-6, EMA 0.05, Sep 5-6 sweep", "On-policy, centring", {0.0: ["g2a5xd86"]}),
    ("SDQL, lr 3e-6, EMA 0.05, Sep 5-6 sweep", "λ=0.9, centring", {0.0: ["7w9cey80"], 1.0: ["msbl1x1p"]}),
    ("SDQL, lr 3e-6, EMA 0.05, Sep 5-6 sweep", "λ=1.0, centring", {0.0: ["eyc3qo1l"], 0.5: ["uf1ioy57"], 1.0: ["zwkexy5h"]}),
    ("SDQL, lr 3e-6, EMA 0.05, Aug 14-18", "On-policy, no centring (Aug 14)", {0.0: ["kgh2r022"], 0.5: ["mfww5t6t"], 1.0: ["7ew1sd0f"]}),
    ("SDQL, lr 3e-6, EMA 0.05, Aug 14-18", "On-policy, centring (Aug 14)", {0.0: ["edipzpdx"], 0.5: ["a9hnpczj"], 1.0: ["j1at7me0"]}),
    ("SDQL, lr 3e-6, EMA 0.05, Aug 14-18", "On-policy, no centring (Aug 17, micro-batch 1)", {0.0: ["syaiqhm1"], 0.5: ["3a50a9ot"], 1.0: ["zm03pq9q"]}),
    ("SDQL, lr 3e-6, EMA 0.05, Aug 14-18", "On-policy, centring (Aug 17, micro-batch 1)", {0.0: ["ly7vex8f"], 0.5: ["y8eq5q0o"], 1.0: ["y9ddre11"]}),
    ("SDQL, lr 3e-6, EMA 0.05, Aug 14-18", "Uniform, no centring (Aug 18)", {0.0: ["xdcs4ryk"], 0.5: ["xwmz3igm"], 1.0: ["t0vtrfiu"]}),
    ("SDQL, lr 3e-6, EMA 0.05, Aug 14-18", "Uniform, centring (Aug 18)", {0.0: ["pauwmhbb"], 0.5: ["t8qss8xb"], 1.0: ["7ztqehbg"]}),
    ("SDQL on-policy + centring, lr 3e-6: teacher rate x mini-batch", "EMA 0, mbs32 (Aug 19)", {0.5: ["tw1l2e9s"], 1.0: ["1fkbd33h"]}),
    ("SDQL on-policy + centring, lr 3e-6: teacher rate x mini-batch", "EMA 0.01, mbs32 (Aug 19)", {0.5: ["llv7pc8y"], 1.0: ["nntcrbf6"]}),
    ("SDQL on-policy + centring, lr 3e-6: teacher rate x mini-batch", "EMA 0, mbs32 (Aug 20)", {0.5: ["12ocxwrk"], 1.0: ["pquoaf1g"]}),
    ("SDQL on-policy + centring, lr 3e-6: teacher rate x mini-batch", "EMA 0.01, mbs32 (Aug 20)", {0.5: ["7jm6nqe7"], 1.0: ["xzn0voxu"]}),
    ("SDQL on-policy + centring, lr 3e-6: teacher rate x mini-batch", "EMA 0.05, mbs32 (Aug 20)", {0.5: ["1m6vtuxy"], 1.0: ["klxawodx"]}),
    ("SDQL on-policy + centring, lr 3e-6: teacher rate x mini-batch", "EMA 0, mbs16", {0.5: ["q9zov1au"], 1.0: ["5mz5a4fb"]}),
    ("SDQL on-policy + centring, lr 3e-6: teacher rate x mini-batch", "EMA 0.01, mbs16", {0.5: ["ooqw4j1s"], 1.0: ["wbl6mkjj"]}),
    ("SDQL on-policy + centring, lr 3e-6: teacher rate x mini-batch", "EMA 0, mbs8", {0.5: ["z8610td1"], 1.0: ["39ws2bjy"]}),
    ("SDQL on-policy + centring, lr 3e-6: teacher rate x mini-batch", "EMA 0.01, mbs8", {0.5: ["9ntvz0tx"], 1.0: ["uj948anc"]}),
]


def best_checkpoint(store, chain):
    """Single checkpoint (step <= MAX_STEP) with the best 4-domain average, and that checkpoint's per-domain values."""
    x, y = store.series(chain, "val")
    if len(y) == 0:
        return None
    k = int(np.argmax(y))
    dom = []
    for metric in DOMAIN_METRICS:
        xd, yd = store.series(chain, metric)
        dom.append(float(yd[list(xd).index(x[k])]))
    return dict(avg4=float(y[k]), step=int(x[k]), domains=dom, last_step=int(x[-1]), last_val=float(y[-1]))


def variant_table(store, out):
    with open(os.path.join(out, "best_checkpoint_per_variant.csv"), "w") as f:
        f.write("group,variant,alpha,avg4,step," + ",".join(DOMAINS) + ",last_step,last_val,runs\n")
        for group, variant, chains in VARIANTS:
            for a, chain in sorted(chains.items()):
                b = best_checkpoint(store, chain)
                if b is None:
                    continue
                f.write(f'"{group}","{variant}",{a},{b["avg4"]:.4f},{b["step"]},' + ",".join(f"{v:.4f}" for v in b["domains"])
                        + f',{b["last_step"]},{b["last_val"]:.4f},{"+".join(chain)}\n')


# --------------------------------------------------------------------------- questions
# Chains list resumed runs in order (same W&B name, later run starts from a checkpoint).

A = [0.0, 0.5, 1.0]


def cols(by_alpha, alphas=A):
    return [(ALPHA_NAME[a], by_alpha[a]) for a in alphas]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--pull", action="store_true")
    args = ap.parse_args()
    if args.pull:
        pull(args.cache)
    store = Store(args.cache)
    out = args.out
    S = []
    SDPO_BASE = {0.0: ["8ovi1uq2"], 0.5: ["54eer5m8"], 1.0: ["1jqglo3j"]}

    # Q1a: env-feedback regime (Aug 8-11): Qwen3-8B, lr 1e-5, is_clip 2, tur 0.05, full-logit top-100.
    q1a = {
        0.0: [["kvd9hund"], ["k2xgjxg6", "lufg2c2e", "gvdsc9b9"], ["3e71h4jo"], ["1ruezpqc"]],
        0.5: [["mq08zmcp"], ["6s046pj5", "w3j4vvg8"], ["epl4sx9x"], ["sd5ltz6h"]],
        1.0: [["oapzokb2", "6i11khkm"], ["g0mwg3bm", "bhpxjjvu"], ["zqa4ysx8", "l3zcepw9"], ["bd8gkfjm"]],
    }
    S += figure_grid(
        store, "Q1a · Does a baseline help REINFORCE + stepwise KL?",
        "With env feedback in the teacher prompt (Aug 8–11 sweep). Qwen3-8B, lr 1e-5, teacher EMA 0.05, IS clip 2.",
        cols(q1a),
        ["SDPO base · stepwise KL only (env feedback)", "+ future returns, no baseline",
         "+ future returns, baseline v1 (Aug 10, pre-fix)", "+ future returns, LOO baseline on G (Aug 11, corrected)"],
        ["val", "critic/score/mean", "actor/grad_norm", "self_distillation/future_kl_token_loss"],
        "q1a_baseline_env_feedback.png", out, refs=references(A),
        notes="LOO baseline = batch-wide, position-wise leave-one-out mean of the return G_t. v1 logged no baseline metric and its PG term "
              "starts uncentred (≈ the no-baseline value), but it still tracks the corrected version on val.",
    )

    # Q1b: no env feedback, rs=remaining, cispo 0.99: group vs batch LOO baseline on r_t, at both teacher rates.
    for tag, tur, spec in [
        ("tur001", "0.01", {0.0: [["pydodnbi"], ["34tib0zu"]], 0.5: [["ckpb6svm"], ["jntmxkfg"]], 1.0: [["7rd3jzcz"], ["dce2eanv"]]}),
        ("tur005", "0.05", {0.0: [["0q2rz5mf"], ["zku4zbt2"]], 0.5: [["7dpnqgxy"], ["26z83fva"]], 1.0: [["mnpamrjs"], ["u6e8sst7"]]}),
    ]:
        S += figure_grid(
            store, "Q1b · Which baseline: per-prompt group vs whole batch?",
            f"No env feedback (Aug 29–30). Future returns with reward centring on r_t, reward scale = remaining, CISPO clip 0.99, teacher EMA {tur}.",
            cols(spec),
            ["+ future returns, group LOO baseline", "+ future returns, batch LOO baseline"],
            ["val", "critic/score/mean", "actor/grad_norm", "self_distillation/future_kl_token_loss"],
            f"q1b_baseline_group_vs_batch_{tag}.png", out, refs=references(A, SDPO_BASE),
            notes="SDPO base exists only at teacher EMA 0.05 (mbs32).",
        )

    # Q2: mean over future rewards: rs none vs remaining, group baseline, cispo 0.99.
    q2_tur05 = {0.0: [["d5cozgd7"], ["0q2rz5mf"]], 0.5: [["wn00dza0"], ["7dpnqgxy"]], 1.0: [["odckkgtf"], ["mnpamrjs"]]}
    q2_tur01 = {0.0: [["0o6f01pm"], ["pydodnbi"]], 0.5: [["nubdqvie"], ["ckpb6svm"]], 1.0: [["wormvd42"], ["7rd3jzcz"]]}
    for tag, spec, tur in [("tur005", q2_tur05, "0.05"), ("tur001", q2_tur01, "0.01")]:
        S += figure_grid(
            store, "Q2 · Sum or mean of future rewards?",
            f"G_t = Σ_(k>t) (r_k − b_k), either summed or divided by the tokens remaining (T − t). Group LOO baseline, CISPO 0.99, teacher EMA {tur}.",
            cols(spec),
            ["Sum (scale = none)", "Mean over remaining tokens (scale = remaining)"],
            ["val", "critic/score/mean", "actor/grad_norm", "self_distillation/future_kl_token_loss"],
            f"q2_reward_scale_{tag}.png", out, refs=references(A, SDPO_BASE),
        )

    # Q3: SDQL without reward centring vs SDPO references.
    q3 = {
        0.0: [["0q2rz5mf"], ["nh85fcx4"], ["x1yt14um"]],
        0.5: [["7dpnqgxy"], ["33b8ef1m"], ["myb1g10y"]],
        1.0: [["mnpamrjs"], ["s0zyvxdm"], ["k41rpp2s"]],
    }
    S += figure_grid(
        store, "Q3 · Does SDQL without reward centring help?",
        "SDQL (rb0 sweep, Sep 5–6, lr 3e-6) against the two SDPO references (lr 1e-5). No env feedback; teacher EMA 0.05 everywhere.",
        cols(q3),
        ["SDPO · + future returns (group LOO, mean)", "SDQL · uniform target, no centring",
         "SDQL · on-policy target, no centring"],
        ["val", "critic/score/mean", "actor/grad_norm"],
        "q3_sdql_no_centering.png", out, refs=references(A, SDPO_BASE),
        notes="Learning rates differ (SDQL 3e-6 vs SDPO 1e-5). Forward-KL SDQL (α=0) diverges in every variant: step-1 loss is around 1e19, "
              "then the policy collapses (val → 0) by step ~60.\n"
              "All runs here use teacher EMA 0.05: SDQL without centring was never run at 0.01 (see Q6 for the teacher-rate effect).",
    )

    # Q4: reward centring on/off within each SDQL target mode (matched pairs from the same sweep).
    q4_rows = [
        ("uniform target", {0.5: [["t8qss8xb"], ["xwmz3igm"]], 1.0: [["7ztqehbg"], ["t0vtrfiu"]]}),
        ("on-policy target", {0.5: [["y8eq5q0o"], ["3a50a9ot"]], 1.0: [["y9ddre11"], ["zm03pq9q"]]}),
        ("λ-return, λ=0.9", {0.5: [[], ["l60ea65a"]], 1.0: [["msbl1x1p"], ["ezh5bdi7"]]}),
        ("λ-return, λ=1.0", {0.5: [["uf1ioy57"], ["14qw269p"]], 1.0: [["zwkexy5h"], ["597ib0aq"]]}),
    ]
    q4_figs = [("val", "q4_sdql_centering_val.png"), ("critic/score/mean", "q4_sdql_centering_train.png")]
    q4_figs += [(f"val/{d}", f"q4_sdql_centering_val_{d}.png") for d in DOMAINS]
    for metric, fname in q4_figs:
        fig, axes = plt.subplots(len(q4_rows), 2, figsize=(9, 2.5 * len(q4_rows) + 1.3), squeeze=False, facecolor=SURFACE)
        labels = ["With reward centring (per-prompt LOO)", "Without reward centring"]
        for i, (rname, spec) in enumerate(q4_rows):
            for j, a in enumerate([0.5, 1.0]):
                ax = axes[i, j]
                plot_panel(ax, store, list(zip(labels, spec[a])), metric, references([a], SDPO_BASE)[0])
                if i == 0:
                    ax.set_title(ALPHA_NAME[a], color=INK, fontsize=10, loc="left")
                if j == 0:
                    ax.set_ylabel(f"{rname}\n{METRIC_LABEL[metric]}", color=INK2, fontsize=8)
                if i == len(q4_rows) - 1:
                    ax.set_xlabel("training step", color=INK2, fontsize=8.5)
                if metric.startswith("val"):
                    chains = spec[a]
                    ends = [val_summary(store, ch)["last_step"] for ch in chains if ch]
                    common = min(ends)
                    for lab, ch in zip(labels, chains):
                        if ch:
                            full, win = val_summary(store, ch, metric=metric), val_summary(store, ch, common, metric=metric)
                            S.append(dict(figure=fname, metric=metric, column=f"{rname} | {ALPHA_NAME[a]}", series=lab, runs=",".join(ch),
                                          common_end=common, **{f"win_{k}": v for k, v in win.items()},
                                          **{f"full_{k}": v for k, v in full.items()}))
        h, l = axes[0, 1].get_legend_handles_labels()
        order = [l.index(x) for x in labels + list(REF_STYLE) if x in l]
        fig.legend([h[k] for k in order], [l[k] for k in order], loc="upper left",
                   bbox_to_anchor=(0.01, 1 - 0.55 / fig.get_figheight()), ncol=3, frameon=False, fontsize=9)
        fig.text(0.01, 1 - 0.12 / fig.get_figheight(), "Q4 · Does reward centring help SDQL?" + (f" · {metric.split('/')[1]} acc/mean@16" if metric.startswith("val/") else ""), fontsize=13, color=INK, weight="bold", va="top")
        fig.text(0.01, 1 - 0.38 / fig.get_figheight(),
                 "Matched pairs from the same sweep. Uniform: Aug 18; on-policy: Aug 17; λ: Sep 6. lr 3e-6, teacher EMA 0.05. α=0 diverges, so it is omitted.",
                 fontsize=8.5, color=INK2, va="top")
        bottom = note_layout(fig, "No centring/no-centring pairs exist at teacher EMA 0.01.\n" + REF_NOTE)
        fig.tight_layout(rect=(0, bottom, 1, 1 - 1.4 / fig.get_figheight()))
        fig.savefig(os.path.join(out, fname), dpi=150, facecolor=SURFACE)
        plt.close(fig)

    # Q5: target modes incl. lambda returns, no centring (rb0 sweep, identical settings).
    q5 = {
        0.5: [["33b8ef1m"], ["myb1g10y"], ["l60ea65a"], ["14qw269p"]],
        1.0: [["s0zyvxdm"], ["k41rpp2s"], ["ezh5bdi7"], ["597ib0aq"]],
    }
    S += figure_grid(
        store, "Q5 · Do λ-returns help SDQL?",
        "rb0 sweep (Sep 5–6): no reward centring, lr 3e-6, teacher EMA 0.05, identical except for the target. α=0 diverges, so it is omitted.",
        cols(q5, [0.5, 1.0]),
        ["Uniform target", "On-policy target", "On-policy λ-return, λ=0.9", "On-policy λ-return, λ=1.0"],
        ["val", "critic/score/mean", "actor/grad_norm", "sdql/target_q_std"],
        "q5_sdql_lambda_returns.png", out, refs=references([0.5, 1.0], SDPO_BASE),
        notes="λ=1.0 at α=0.5 has non-finite gradients in 305 of 400 steps (skipped updates), so its policy is effectively frozen after step ~85.\n"
              "Grad clip is 1.0, so every SDQL step is clipped. Teacher EMA 0.05: these targets were never run at 0.01.",
    )

    # Q6: teacher update rate (EMA) for future returns and SDQL.
    q6_fr = {
        0.0: [["0q2rz5mf"], ["pydodnbi"], ["zku4zbt2"], ["34tib0zu"]],
        0.5: [["7dpnqgxy"], ["ckpb6svm"], ["26z83fva"], ["jntmxkfg"]],
        1.0: [["mnpamrjs"], ["7rd3jzcz"], ["u6e8sst7"], ["dce2eanv"]],
    }
    S += figure_grid(
        store, "Q6a · Does a slower teacher (EMA 0.01) help future returns?",
        "SDPO + future returns, reward scale = remaining, CISPO 0.99, no env feedback (Aug 29–30). Only the teacher update rate differs within a colour pair.",
        cols(q6_fr),
        ["Group LOO · teacher EMA 0.05", "Group LOO · teacher EMA 0.01", "Batch LOO · teacher EMA 0.05", "Batch LOO · teacher EMA 0.01"],
        ["val", "critic/score/mean", "actor/grad_norm"],
        "q6a_teacher_rate_future_returns.png", out, refs=references(A, SDPO_BASE),
    )
    q6_sdql = {
        0.5: [["12ocxwrk"], ["7jm6nqe7"], ["1m6vtuxy"]],
        1.0: [["pquoaf1g"], ["xzn0voxu"], ["klxawodx"]],
    }
    S += figure_grid(
        store, "Q6b · Does a slower teacher help SDQL?",
        "SDQL, on-policy target with reward centring (fx2 sweep, Aug 20), lr 3e-6. Only the teacher update rate differs. 0 = frozen teacher.",
        cols(q6_sdql, [0.5, 1.0]),
        ["Teacher EMA 0 (frozen)", "Teacher EMA 0.01", "Teacher EMA 0.05"],
        ["val", "critic/score/mean", "sdql/target_q_std"],
        "q6b_teacher_rate_sdql.png", out, refs=references([0.5, 1.0], SDPO_BASE),
        notes="This is the only SDQL configuration swept over teacher rate at mbs32 (plus the Aug-19 'tur' runs, which agree).",
    )

    # Q7: SDQL mini-batch size (more optimizer updates per rollout batch at the same wall-clock).
    for tag, tur, spec in [
        ("tur001", "0.01", {0.5: [["7jm6nqe7"], ["ooqw4j1s"], ["9ntvz0tx"]], 1.0: [["xzn0voxu"], ["wbl6mkjj"], ["uj948anc"]]}),
        ("tur0", "0 (frozen)", {0.5: [["12ocxwrk"], ["q9zov1au"], ["z8610td1"]], 1.0: [["pquoaf1g"], ["5mz5a4fb"], ["39ws2bjy"]]}),
    ]:
        S += figure_grid(
            store, "Q7 · Does a smaller mini-batch help SDQL?",
            f"SDQL, on-policy + centring, lr 3e-6, teacher EMA {tur}. Only the mini-batch differs (fx2 mbs32 vs ofp mbs16/8).",
            cols(spec, [0.5, 1.0]),
            ["Mini-batch 32 (1 update / step)", "Mini-batch 16 (2 updates / step)", "Mini-batch 8 (4 updates / step)"],
            ["val", "critic/score/mean", "sdql/target_q_std", "timing_s/step"],
            f"q7_sdql_minibatch_{tag}.png", out, refs=references([0.5, 1.0], SDPO_BASE),
            notes="Train batch is 32 prompts × 8 rollouts in every run, so each step costs the same (~88 s); a smaller mini-batch only "
                  "splits it into more optimizer updates.",
        )

    # Overview: best run per family at its best teacher rate (mean val over steps <= 400).
    fam = {
        0.0: [["pydodnbi"], [], []],
        0.5: [["jntmxkfg"], ["1m6vtuxy"], ["9ntvz0tx"]],
        1.0: [["7rd3jzcz"], ["xzn0voxu"], ["wbl6mkjj"]],
    }
    S += figure_grid(
        store, "Overview · Best variant of each family vs references",
        "Best run per family by mean val acc over steps ≤ 400, at its best teacher rate. No env feedback.",
        cols(fam),
        ["SDPO + future returns (best: EMA 0.01; group at α=0,1, batch at α=0.5)", "SDQL mbs32 (best: on-policy + centring; EMA 0.05 at α=0.5, 0.01 at α=1)",
         "SDQL mbs 8/16 (best: on-policy + centring, EMA 0.01; mbs8 at α=0.5, mbs16 at α=1)"],
        ["val", "critic/score/mean"],
        "q0_overview.png", out, refs=references(A, SDPO_BASE),
        notes="SDQL at α=0 (forward KL) diverges in every variant, so that panel has no SDQL line. "
              "SDPO lines are mbs32 only: no matching SDPO runs exist at mbs 8/16.",
    )

    best_tables(store, out)
    variant_table(store, out)

    json.dump(S, open(os.path.join(out, "summary.json"), "w"), indent=1)
    with open(os.path.join(out, "summary.csv"), "w") as f:
        keys = ["figure", "metric", "column", "series", "runs", "common_end", "win_final", "win_mean", "win_best", "full_last_step",
                "full_final", "full_mean", "full_best"]
        f.write(",".join(keys) + "\n")
        for r in S:
            f.write(",".join(f'"{r.get(k, "")}"' if isinstance(r.get(k), str) else
                             (f"{r[k]:.4f}" if isinstance(r.get(k), float) else str(r.get(k, ""))) for k in keys) + "\n")
    print(f"wrote {len(S)} summary rows to {out}")


if __name__ == "__main__":
    main()
