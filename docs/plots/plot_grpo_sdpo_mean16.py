"""Export and plot Qwen3-8B GRPO versus SDPO validation mean@16."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb


OUT_DIR = Path(__file__).resolve().parent
RUNS = {
    "GRPO": "harshg5799/SDPO2-hrxu/kwe6y1fy",
    "SDPO": "harshg5799/SDPO2-hrxu/mq08zmcp",
}
MAX_STEP = 300
DOMAINS = ("biology", "chemistry", "material", "physics")
METRICS = {
    domain: f"val-core/sciknoweval_{domain}/acc/mean@16" for domain in DOMAINS
}


def main():
    api = wandb.Api(timeout=30)
    records = []
    for method, path in RUNS.items():
        run = api.run(path)
        keys = ["training/global_step", *METRICS.values()]
        history = [
            row
            for row in run.scan_history(keys=keys, page_size=1000)
            if int(row["training/global_step"]) <= MAX_STEP
        ]
        if not history:
            raise RuntimeError(f"No complete validation rows for {path}")
        for row in history:
            for domain, key in METRICS.items():
                records.append(
                    {
                        "method": method,
                        "run_path": path,
                        "training_step": int(row["training/global_step"]),
                        "domain": domain,
                        "mean_at_16": float(row[key]),
                    }
                )

    csv_path = OUT_DIR / "grpo_sdpo_mean16.csv"
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(9, 12), sharex=True, constrained_layout=True)
    colors = {"GRPO": "#2b6cb0", "SDPO": "#d95f02"}
    for ax, domain in zip(axes.flat, DOMAINS):
        domain_values = [r["mean_at_16"] for r in records if r["domain"] == domain]
        for method in RUNS:
            points = sorted(
                (r for r in records if r["domain"] == domain and r["method"] == method),
                key=lambda r: r["training_step"],
            )
            ax.plot(
                [p["training_step"] for p in points],
                [100 * p["mean_at_16"] for p in points],
                marker="o",
                linewidth=2,
                markersize=4,
                color=colors[method],
                label=method,
            )
        ax.set_title(domain.capitalize())
        ax.set_xlim(0, MAX_STEP)
        ax.set_ylim(100 * (min(domain_values) - 0.1), 100 * (max(domain_values) + 0.1))
        ax.grid(alpha=0.2)
    fig.supxlabel("Training step")
    fig.supylabel("Validation accuracy, mean@16 (%)")
    fig.suptitle("Qwen3-8B: GRPO vs SDPO on SciKnowEval")
    fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 0.95), ncol=2, frameon=False)
    fig.savefig(OUT_DIR / "grpo_sdpo_mean16.png", dpi=180)
    plt.close(fig)
    print(f"Wrote {csv_path} and {OUT_DIR / 'grpo_sdpo_mean16.png'}")


if __name__ == "__main__":
    main()
