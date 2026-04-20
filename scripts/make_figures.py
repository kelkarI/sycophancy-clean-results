"""Generate fig1 (Δlogit paired bar), fig2 (Δrate paired bar),
fig3 (per-seed dot plot), fig4 (cosine heatmap, one per model).

Reads from data/*_clean.json produced by scripts/build_data.py.
Writes to figures/*.{pdf,png}.

Deterministic: no RNG. Plot order fixed by _style.DEFAULT_ORDER.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _style import (
    PALETTE, LABELS, CRITICAL, CONFORMIST, DEFAULT_ORDER,
    sig_marker, save,
)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FIG  = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

MODEL_FILES = {
    "Gemma 2 27B": DATA / "gemma-2-27b-it_clean.json",
    "Qwen 3 32B":  DATA / "qwen3-32b_clean.json",
}
COS_FILES = {
    "Gemma 2 27B": DATA / "gemma-2-27b-it_cosines.json",
    "Qwen 3 32B":  DATA / "qwen3-32b_cosines.json",
}


def load(p):
    return json.loads(Path(p).read_text())


def bar_panel(ax, block, metric, ylabel):
    """Single-model bar panel. metric in {'delta_logit', 'delta_rate_pp'}."""
    conds = block["conditions"]
    xs, ys, yerr_lo, yerr_hi, colors, sig_txt = [], [], [], [], [], []
    for i, c in enumerate(DEFAULT_ORDER):
        cd = conds[c]
        if metric == "delta_logit":
            m  = cd["delta_logit_mean"]
            lo = cd["delta_logit_ci95_lo"]
            hi = cd["delta_logit_ci95_hi"]
        else:
            m  = cd["delta_rate_mean_pp"]
            lo = cd["delta_rate_ci95_lo_pp"]
            hi = cd["delta_rate_ci95_hi_pp"]
        xs.append(i); ys.append(m)
        yerr_lo.append(max(0.0, m - lo)); yerr_hi.append(max(0.0, hi - m))
        colors.append(PALETTE[c])
        if c == "random":
            sig_txt.append("")
        else:
            sig_txt.append(sig_marker(cd.get("n_significant_after_holm_per_seed")))
    ax.axhline(0, color="black", lw=0.8, zorder=1)
    ax.bar(xs, ys, yerr=[yerr_lo, yerr_hi],
           color=colors, edgecolor="black", linewidth=0.5,
           error_kw=dict(ecolor="black", capsize=3, lw=0.8), zorder=2)
    # Label asterisks above the bar (or below if the bar is positive)
    for x, y, mark in zip(xs, ys, sig_txt):
        if not mark:
            continue
        va = "top" if y > 0 else "bottom"
        ypos = y - 0.02 if y > 0 else y + 0.02
        ax.text(x, ypos, mark, ha="center", va=va, fontsize=11)
    ax.set_xticks(xs)
    ax.set_xticklabels([LABELS[c] for c in DEFAULT_ORDER],
                       rotation=28, ha="right")
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3, linestyle=":")


def fig1_delta_logit():
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=False)
    for ax, (title, p) in zip(axes, MODEL_FILES.items()):
        block = load(p)
        bar_panel(ax, block, "delta_logit",
                  r"$\Delta$ sycophancy logit" "\n(negative = steering reduces syc)")
        baseline_logit = block["baseline_mean_logit"]
        ax.set_title(f"{title}  (baseline logit = {baseline_logit:+.2f})")
    fig.suptitle(
        "Sycophancy-logit change at each condition's tune-locked "
        "coefficient\n(mean over 3 test seeds; error bars 95% CI; "
        "*** = Holm-sig in 3/3 seeds, ** in 2/3, * in 1/3)",
        fontsize=10,
    )
    plt.tight_layout()
    save(fig, str(FIG / "fig1_delta_logit"))
    plt.close(fig)


def fig2_delta_rate():
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=False)
    for ax, (title, p) in zip(axes, MODEL_FILES.items()):
        block = load(p)
        bar_panel(ax, block, "delta_rate_pp",
                  r"$\Delta$ sycophancy rate (pp)" "\n(negative = steering reduces syc)")
        baseline_rate = block["baseline_mean_rate"] * 100
        ax.set_title(f"{title}  (baseline rate = {baseline_rate:.1f}%)")
    fig.suptitle(
        "Binary sycophancy rate change at each condition's tune-locked "
        "coefficient\n(mean over 3 test seeds; error bars 95% CI; "
        "asterisks as in Fig 1, based on paired Wilcoxon on logits)",
        fontsize=10,
    )
    plt.tight_layout()
    save(fig, str(FIG / "fig2_delta_rate"))
    plt.close(fig)


def fig3_per_seed_dots():
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0), sharey=False)
    for ax, (title, p) in zip(axes, MODEL_FILES.items()):
        block = load(p)
        conds = block["conditions"]
        x_positions = {}
        for i, c in enumerate(DEFAULT_ORDER):
            x_positions[c] = i
        # Light jitter: three fixed offsets so dots don't overlap.
        offsets = [-0.18, 0.0, 0.18]
        for c in DEFAULT_ORDER:
            cd = conds[c]
            # random's per_seed holds mean_delta_logit; real conds hold delta_logit
            if c == "random":
                vals = [p["mean_delta_logit"] for p in cd["per_seed"]]
            else:
                vals = [p["delta_logit"] for p in cd["per_seed"]]
            for off, v in zip(offsets, vals):
                ax.plot(x_positions[c] + off, v, "o",
                        color=PALETTE[c], markersize=7,
                        markeredgecolor="black", markeredgewidth=0.6,
                        zorder=3)
            # horizontal bracket for the mean
            m = np.mean(vals)
            ax.plot([x_positions[c] - 0.25, x_positions[c] + 0.25], [m, m],
                    color="black", lw=1.5, zorder=4)
        ax.axhline(0, color="black", lw=0.8, zorder=1)
        ax.set_xticks(list(x_positions.values()))
        ax.set_xticklabels([LABELS[c] for c in DEFAULT_ORDER],
                           rotation=28, ha="right")
        ax.set_ylabel(r"$\Delta$ sycophancy logit (per seed)")
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3, linestyle=":")
    fig.suptitle(
        "Per-seed consistency (3 test seeds: 42, 7, 123; horizontal "
        "bars mark the per-condition mean)", fontsize=10,
    )
    plt.tight_layout()
    save(fig, str(FIG / "fig3_per_seed"))
    plt.close(fig)


def fig4_cosines():
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0))
    for ax, (title, p) in zip(axes, COS_FILES.items()):
        c = load(p)
        names = [LABELS.get(n, n) for n in c["names"]]
        mat = np.array(c["matrix"])
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-0.5, vmax=1.0)
        ax.set_xticks(range(len(names)))
        ax.set_yticks(range(len(names)))
        ax.set_xticklabels(names, rotation=30, ha="right")
        ax.set_yticklabels(names)
        for i in range(len(names)):
            for j in range(len(names)):
                v = mat[i, j]
                col = "white" if abs(v) > 0.6 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color=col, fontsize=8)
        ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                     label="cosine similarity")
    fig.suptitle("Pairwise cosine similarity between kept steering "
                 "vectors (at the target layer of each model)",
                 fontsize=10)
    plt.tight_layout()
    save(fig, str(FIG / "fig4_cosines"))
    plt.close(fig)


def main():
    print("Building figures...")
    fig1_delta_logit(); print("  saved fig1_delta_logit.{pdf,png}")
    fig2_delta_rate();  print("  saved fig2_delta_rate.{pdf,png}")
    fig3_per_seed_dots(); print("  saved fig3_per_seed.{pdf,png}")
    fig4_cosines();     print("  saved fig4_cosines.{pdf,png}")
    print(f"All figures in {FIG}")


if __name__ == "__main__":
    main()
