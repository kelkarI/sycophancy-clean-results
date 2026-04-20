"""Generate fig6 (filtered steering-coefficient sweeps, one panel per model).

The source pipelines (sycophancy-gemma, sycophancy-qwen) each produce a
`fig1_steering_curves.png` showing sycophancy-rate and sycophancy-logit
versus the steering coefficient for every condition in CONDITIONS_REAL
(assistant_axis, 5 critical roles, 4 conformist roles, CAA, 3 residual
conditions) plus the 10 random controls. With 10+ overlapping lines the
plot is unreadable. This script renders the same sweep but filtered to
the kept subset (see README.md "Scope"):

  critical   : skeptic, devils_advocate, judge
  conformist : peacekeeper, pacifist, collaborator
  targeted   : caa
  null       : random mean (n=10) with +/- std band
  reference  : baseline (coef = 0)

Reads `sycophancy_rates_test.json` from the source repos (same files the
upstream fig1 scripts use), so the plotted values are identical to the
parent pipeline up to the filtered condition set.

Deterministic: no RNG. Legend order is fixed (CAA, critical, conformist,
random, baseline).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _style import PALETTE, LABELS, CRITICAL, CONFORMIST, save

ROOT = Path(__file__).resolve().parent.parent
FIG  = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# Source rates files. Override via env vars in build_data.py style if the
# source repos live elsewhere (same convention as build_data.py).
GEMMA_RATES = Path(
    "/lambda/nfs/filesystem/sycophancy-gemma/experiment-main/results/"
    "sycophancy_rates_test.json"
)
QWEN_RATES = Path(
    "/lambda/nfs/filesystem/sycophancy-qwen/results/sycophancy_rates_test.json"
)

# Fall back to relative sibling paths if the absolute ones are missing
# (lets this run inside a worktree or local clone).
_REL_GEMMA = ROOT.parent / "sycophancy-gemma/experiment-main/results/sycophancy_rates_test.json"
_REL_QWEN  = ROOT.parent / "sycophancy-qwen/results/sycophancy_rates_test.json"
if not GEMMA_RATES.exists() and _REL_GEMMA.exists():
    GEMMA_RATES = _REL_GEMMA
if not QWEN_RATES.exists() and _REL_QWEN.exists():
    QWEN_RATES = _REL_QWEN

MODEL_SOURCES = [
    ("Gemma 2 27B", GEMMA_RATES),
    ("Qwen 3 32B",  QWEN_RATES),
]

KEPT_ORDER = ["caa"] + CRITICAL + CONFORMIST
N_RANDOM   = 10
RANDOM_CONDS = [f"random_{i}" for i in range(N_RANDOM)]


def load_rates(path):
    return json.loads(Path(path).read_text())


def coeffs_of(rates, cond):
    """Sorted numeric coefficients for a condition."""
    return sorted((float(k) for k in rates[cond]), key=lambda x: x)


def _baseline_logit_rate(rates):
    """Baseline = coef=0 on any real condition (identical across conds)."""
    cond = next(c for c in rates if not c.startswith("random_"))
    cell = rates[cond]["0.0"]
    return float(cell["mean_syc_logit"]), float(cell["binary_rate"])


def _series(rates, cond, coeffs, metric):
    ys = []
    for c in coeffs:
        k = f"{c}" if c != 0.0 else "0.0"
        cell = rates[cond].get(k)
        if cell is None:
            ys.append(np.nan)
            continue
        if metric == "rate":
            ys.append(float(cell["binary_rate"]) * 100)
        else:
            ys.append(float(cell["mean_syc_logit"]))
    return ys


def _random_band(rates, coeffs, metric):
    """Mean and std across random_0..9 at each coefficient."""
    means, stds = [], []
    for c in coeffs:
        k = f"{c}" if c != 0.0 else "0.0"
        vals = []
        for rc in RANDOM_CONDS:
            cell = rates.get(rc, {}).get(k)
            if cell is None:
                continue
            if metric == "rate":
                vals.append(float(cell["binary_rate"]) * 100)
            else:
                vals.append(float(cell["mean_syc_logit"]))
        if vals:
            means.append(float(np.mean(vals)))
            stds.append(float(np.std(vals)))
        else:
            means.append(np.nan)
            stds.append(0.0)
    return means, stds


def _plot_one(ax, rates, metric, title, ylabel, put_legend):
    cond_any = next(c for c in rates if not c.startswith("random_"))
    coeffs = coeffs_of(rates, cond_any)
    for c in KEPT_ORDER:
        if c not in rates:
            continue
        ys = _series(rates, c, coeffs, metric)
        ax.plot(coeffs, ys, color=PALETTE[c], marker="o", markersize=3.5,
                lw=1.6, label=LABELS[c])
    rm, rs = _random_band(rates, coeffs, metric)
    ax.plot(coeffs, rm, color=PALETTE["random"], ls="--", marker="s",
            markersize=3, lw=1.4, label=f"Random mean (n={N_RANDOM})")
    lo = [m - s for m, s in zip(rm, rs)]
    hi = [m + s for m, s in zip(rm, rs)]
    ax.fill_between(coeffs, lo, hi, color=PALETTE["random"], alpha=0.22,
                    linewidth=0)
    bl_logit, bl_rate = _baseline_logit_rate(rates)
    ax.axhline(bl_rate * 100 if metric == "rate" else bl_logit,
               ls=":", color="black", alpha=0.8, lw=1.0,
               label=f"baseline = "
                     + (f"{bl_rate*100:.1f}%" if metric == "rate"
                        else f"{bl_logit:+.2f}"))
    ax.set_xlabel("Steering coefficient")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3, linestyle=":")
    if put_legend:
        ax.legend(loc="best", fontsize=8, ncol=2, framealpha=0.9)


def make_figure():
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5))
    for col, (model_title, rates_path) in enumerate(MODEL_SOURCES):
        rates = load_rates(rates_path)
        _plot_one(axes[0, col], rates, "rate",
                  f"{model_title}: sycophancy rate",
                  "Sycophancy rate (%)",
                  put_legend=(col == 0))
        _plot_one(axes[1, col], rates, "logit",
                  f"{model_title}: mean sycophancy logit",
                  r"Mean sycophancy logit  $\log p(\mathrm{syc}) - \log p(\mathrm{hon})$",
                  put_legend=False)
    fig.suptitle(
        "Sycophancy response to steering coefficient (kept conditions only)\n"
        "rows = metric; columns = model; shaded band = random ctrl mean +/- std",
        fontsize=11,
    )
    plt.tight_layout()
    save(fig, str(FIG / "fig6_steering_curves"))
    plt.close(fig)


def main():
    print("Building fig6_steering_curves...")
    make_figure()
    print(f"  saved fig6_steering_curves.{{pdf,png}} in {FIG}")


if __name__ == "__main__":
    main()
