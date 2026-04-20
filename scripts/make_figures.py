"""Generate fig1 (Δlogit paired bar), fig2 (Δrate paired bar),
fig3 (per-seed dot plot), fig4 (cosine heatmap, one per model).

Core figures keep every condition and hatch the bars/dots for any
condition flagged as degraded at its tune-locked coefficient in the
source pipeline (a degraded cell is model collapse — uniform logits,
binary rate ≈ 0.5 — not a sycophancy reduction).

Filtered figures (`fig1_filtered`, `fig2_filtered`, `fig3_filtered`)
drop degraded conditions entirely so the plot reflects only cells
where the steered forward pass is behaving normally.

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


def bar_panel(ax, block, metric, ylabel, drop_degraded=False):
    """Single-model bar panel. metric in {'delta_logit', 'delta_rate_pp'}.

    If drop_degraded is False, degraded cells are plotted with a hatch
    overlay and annotated "degraded" below their asterisks. If True,
    the cell is skipped entirely and its x slot is left empty.
    """
    conds = block["conditions"]
    order = [c for c in DEFAULT_ORDER
             if not (drop_degraded and conds[c].get("degraded_any_seed"))]
    xs, ys, yerr_lo, yerr_hi, colors, sig_txt, degraded = (
        [], [], [], [], [], [], []
    )
    for i, c in enumerate(order):
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
        degraded.append(bool(cd.get("degraded_any_seed")))
        if c == "random":
            sig_txt.append("")
        else:
            sig_txt.append(sig_marker(cd.get("n_significant_after_holm_per_seed")))
    ax.axhline(0, color="black", lw=0.8, zorder=1)
    hatches = ["//" if d else "" for d in degraded]
    bars = ax.bar(xs, ys, yerr=[yerr_lo, yerr_hi],
                  color=colors, edgecolor="black", linewidth=0.5,
                  error_kw=dict(ecolor="black", capsize=3, lw=0.8),
                  zorder=2)
    for bar, h in zip(bars, hatches):
        if h:
            bar.set_hatch(h)
    for x, y, mark, deg in zip(xs, ys, sig_txt, degraded):
        if mark:
            va = "top" if y > 0 else "bottom"
            ypos = y - 0.02 if y > 0 else y + 0.02
            ax.text(x, ypos, mark, ha="center", va=va, fontsize=11)
        if deg:
            va = "bottom" if y > 0 else "top"
            ypos = y + 0.06 if y > 0 else y - 0.06
            ax.text(x, ypos, "degraded", ha="center", va=va,
                    fontsize=7, color="#8b0000",
                    rotation=90 if ylabel.startswith(r"$\Delta$ sycophancy logit") else 0)
    ax.set_xticks(xs)
    ax.set_xticklabels([LABELS[c] for c in order],
                       rotation=28, ha="right")
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3, linestyle=":")


def _fig1_render(drop_degraded, out_stem):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=False)
    for ax, (title, p) in zip(axes, MODEL_FILES.items()):
        block = load(p)
        bar_panel(ax, block, "delta_logit",
                  r"$\Delta$ sycophancy logit" "\n(negative = steering reduces syc)",
                  drop_degraded=drop_degraded)
        baseline_logit = block["baseline_mean_logit"]
        ax.set_title(f"{title}  (baseline logit = {baseline_logit:+.2f})")
    suffix = (" — degradation-filtered (collapsed cells removed)"
              if drop_degraded else "")
    fig.suptitle(
        "Sycophancy-logit change at each condition's tune-locked "
        "coefficient" + suffix +
        "\n(mean over 3 test seeds; error bars 95% CI; "
        "*** = Holm-sig in 3/3 seeds, ** in 2/3, * in 1/3; "
        "hatched bar = degraded)",
        fontsize=10,
    )
    plt.tight_layout()
    save(fig, str(FIG / out_stem))
    plt.close(fig)


def fig1_delta_logit():
    _fig1_render(drop_degraded=False, out_stem="fig1_delta_logit")
    _fig1_render(drop_degraded=True,  out_stem="fig1_delta_logit_filtered")


def _fig2_render(drop_degraded, out_stem):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=False)
    for ax, (title, p) in zip(axes, MODEL_FILES.items()):
        block = load(p)
        bar_panel(ax, block, "delta_rate_pp",
                  r"$\Delta$ sycophancy rate (pp)" "\n(negative = steering reduces syc)",
                  drop_degraded=drop_degraded)
        baseline_rate = block["baseline_mean_rate"] * 100
        ax.set_title(f"{title}  (baseline rate = {baseline_rate:.1f}%)")
    suffix = (" — degradation-filtered (collapsed cells removed)"
              if drop_degraded else "")
    fig.suptitle(
        "Binary sycophancy rate change at each condition's tune-locked "
        "coefficient" + suffix +
        "\n(mean over 3 test seeds; error bars 95% CI; "
        "asterisks as in Fig 1, based on paired Wilcoxon on logits; "
        "hatched bar = degraded)",
        fontsize=10,
    )
    plt.tight_layout()
    save(fig, str(FIG / out_stem))
    plt.close(fig)


def fig2_delta_rate():
    _fig2_render(drop_degraded=False, out_stem="fig2_delta_rate")
    _fig2_render(drop_degraded=True,  out_stem="fig2_delta_rate_filtered")


def _fig3_render(drop_degraded, out_stem):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0), sharey=False)
    for ax, (title, p) in zip(axes, MODEL_FILES.items()):
        block = load(p)
        conds = block["conditions"]
        order = [c for c in DEFAULT_ORDER
                 if not (drop_degraded and conds[c].get("degraded_any_seed"))]
        x_positions = {c: i for i, c in enumerate(order)}
        offsets = [-0.18, 0.0, 0.18]
        for c in order:
            cd = conds[c]
            if c == "random":
                vals = [p["mean_delta_logit"] for p in cd["per_seed"]]
            else:
                vals = [p["delta_logit"] for p in cd["per_seed"]]
            face = PALETTE[c]
            edge = "black"
            # In the core figure, mark degraded dots with an X overlay.
            deg = bool(cd.get("degraded_any_seed"))
            for off, v in zip(offsets, vals):
                ax.plot(x_positions[c] + off, v, "o",
                        color=face, markersize=7,
                        markeredgecolor=edge, markeredgewidth=0.6,
                        zorder=3)
                if deg and not drop_degraded:
                    ax.plot(x_positions[c] + off, v, "x",
                            color="#8b0000", markersize=6,
                            markeredgewidth=1.5, zorder=4)
            m = np.mean(vals)
            ax.plot([x_positions[c] - 0.25, x_positions[c] + 0.25], [m, m],
                    color="black", lw=1.5, zorder=4)
            if deg and not drop_degraded:
                y_annot = max(vals) + 0.15 if m < 0 else min(vals) - 0.15
                ax.text(x_positions[c], y_annot, "degraded",
                        ha="center",
                        va="bottom" if m < 0 else "top",
                        fontsize=7, color="#8b0000")
        ax.axhline(0, color="black", lw=0.8, zorder=1)
        ax.set_xticks(list(x_positions.values()))
        ax.set_xticklabels([LABELS[c] for c in order],
                           rotation=28, ha="right")
        ax.set_ylabel(r"$\Delta$ sycophancy logit (per seed)")
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3, linestyle=":")
    suffix = (" — degradation-filtered" if drop_degraded else "")
    fig.suptitle(
        "Per-seed consistency (3 test seeds: 42, 7, 123; horizontal bars "
        "mark the per-condition mean" +
        ("; red × marks degraded dots)" if not drop_degraded else ")") + suffix,
        fontsize=10,
    )
    plt.tight_layout()
    save(fig, str(FIG / out_stem))
    plt.close(fig)


def fig3_per_seed_dots():
    _fig3_render(drop_degraded=False, out_stem="fig3_per_seed")
    _fig3_render(drop_degraded=True,  out_stem="fig3_per_seed_filtered")


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
    fig1_delta_logit()
    print("  saved fig1_delta_logit{,_filtered}.{pdf,png}")
    fig2_delta_rate()
    print("  saved fig2_delta_rate{,_filtered}.{pdf,png}")
    fig3_per_seed_dots()
    print("  saved fig3_per_seed{,_filtered}.{pdf,png}")
    fig4_cosines()
    print("  saved fig4_cosines.{pdf,png}")
    print(f"All figures in {FIG}")


if __name__ == "__main__":
    main()
