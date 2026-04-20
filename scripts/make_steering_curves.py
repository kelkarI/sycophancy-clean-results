"""Generate fig6 (per-role sweep) and fig7 (family-mean sweep).

The source pipelines (sycophancy-gemma, sycophancy-qwen) each produce a
`fig1_steering_curves.png` showing sycophancy-rate and sycophancy-logit
versus the steering coefficient for every condition in CONDITIONS_REAL
(assistant_axis, 5 critical roles, 4 conformist roles, CAA, 3 residual
conditions) plus the 10 random controls. With 10+ overlapping lines the
plot is unreadable. This script renders two cleaned views:

  fig6_steering_curves            -- one line per kept condition
  fig7_steering_curves_family     -- three lines per model (CAA + the
                                     critical family mean + the
                                     conformist family mean), with a
                                     min/max band around each family
                                     line. Degraded cells are masked
                                     before averaging.

Kept conditions (see README.md "Scope"):

  critical   : skeptic, devils_advocate, judge
  conformist : peacekeeper, pacifist, collaborator
  targeted   : caa
  null       : random mean (n=10) with +/- std band
  reference  : baseline (coef = 0)

Reads `sycophancy_rates_test.json` and `degradation_flags_test.json`
from the source repos (same files the upstream fig1 scripts use), so
the plotted values are identical to the parent pipeline up to the
filtered condition set.

See README.md section "Family averaging (fig7)" for the exact procedure
used to aggregate the per-role curves into a family-mean curve.

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

# Source result directories. Override via env vars in build_data.py style
# if the source repos live elsewhere (same convention as build_data.py).
GEMMA_RES = Path(
    "/lambda/nfs/filesystem/sycophancy-gemma/experiment-main/results"
)
QWEN_RES = Path("/lambda/nfs/filesystem/sycophancy-qwen/results")

# Fall back to relative sibling paths if the absolute ones are missing
# (lets this run inside a worktree or local clone).
_REL_GEMMA = ROOT.parent / "sycophancy-gemma/experiment-main/results"
_REL_QWEN  = ROOT.parent / "sycophancy-qwen/results"
if not GEMMA_RES.exists() and _REL_GEMMA.exists():
    GEMMA_RES = _REL_GEMMA
if not QWEN_RES.exists() and _REL_QWEN.exists():
    QWEN_RES = _REL_QWEN

MODEL_SOURCES = [
    ("Gemma 2 27B", GEMMA_RES),
    ("Qwen 3 32B",  QWEN_RES),
]

KEPT_ORDER = ["caa"] + CRITICAL + CONFORMIST
N_RANDOM   = 10
RANDOM_CONDS = [f"random_{i}" for i in range(N_RANDOM)]


def load_rates(res_dir):
    return json.loads((Path(res_dir) / "sycophancy_rates_test.json").read_text())


def load_degraded(res_dir):
    """Return dict[cond][coef_key] -> bool. coef_key matches rates keys.

    Top-level `degradation_flags_test.json` is an OR across test seeds
    (any seed flagged -> True). That matches the degraded_any_seed
    convention used in data/*_clean.json.
    """
    return json.loads((Path(res_dir) / "degradation_flags_test.json").read_text())


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


def _family_series(rates, degraded, members, coeffs, metric):
    """Family mean + min/max across members at each coefficient.

    Masking rule (see README "Family averaging (fig7)"):
      at each coefficient c, only include role r in the mean/band if
      `degraded[r][c]` is False. If zero members survive, that point is
      NaN and is not plotted (matplotlib breaks the line there).
    """
    means, lo, hi = [], [], []
    for c in coeffs:
        k = f"{c}" if c != 0.0 else "0.0"
        vals = []
        for r in members:
            if degraded.get(r, {}).get(k, False):
                continue
            cell = rates.get(r, {}).get(k)
            if cell is None:
                continue
            if metric == "rate":
                vals.append(float(cell["binary_rate"]) * 100)
            else:
                vals.append(float(cell["mean_syc_logit"]))
        if vals:
            means.append(float(np.mean(vals)))
            lo.append(float(np.min(vals)))
            hi.append(float(np.max(vals)))
        else:
            means.append(np.nan); lo.append(np.nan); hi.append(np.nan)
    return means, lo, hi


def _plot_family(ax, rates, degraded, metric, title, ylabel, put_legend):
    cond_any = next(c for c in rates if not c.startswith("random_"))
    coeffs = coeffs_of(rates, cond_any)

    # CAA -- single member, no band.
    caa_ys = _series(rates, "caa", coeffs, metric)
    # Drop degraded CAA points (NaN them so the line breaks there).
    caa_ys_masked = [
        np.nan if degraded.get("caa", {}).get(
            f"{c}" if c != 0.0 else "0.0", False) else y
        for c, y in zip(coeffs, caa_ys)
    ]
    ax.plot(coeffs, caa_ys_masked, color=PALETTE["caa"], marker="D",
            markersize=4, lw=1.8, label=LABELS["caa"])

    # Critical family -- mean line + min/max band.
    for family, members, color_key, label in [
        ("critical",   CRITICAL,   "skeptic",     "Critical mean (n=3)"),
        ("conformist", CONFORMIST, "peacekeeper", "Conformist mean (n=3)"),
    ]:
        m, lo, hi = _family_series(rates, degraded, members, coeffs, metric)
        col = PALETTE[color_key]
        ax.plot(coeffs, m, color=col, marker="o", markersize=4, lw=1.9,
                label=label)
        ax.fill_between(
            coeffs,
            [np.nan if np.isnan(x) else x for x in lo],
            [np.nan if np.isnan(x) else x for x in hi],
            color=col, alpha=0.22, linewidth=0,
        )

    # Random null band -- mean +/- std as before.
    rm, rs = _random_band(rates, coeffs, metric)
    ax.plot(coeffs, rm, color=PALETTE["random"], ls="--", marker="s",
            markersize=3, lw=1.4, label=f"Random mean (n={N_RANDOM})")
    ax.fill_between(coeffs,
                    [m - s for m, s in zip(rm, rs)],
                    [m + s for m, s in zip(rm, rs)],
                    color=PALETTE["random"], alpha=0.18, linewidth=0)

    bl_logit, bl_rate = _baseline_logit_rate(rates)
    ax.axhline(bl_rate * 100 if metric == "rate" else bl_logit,
               ls=":", color="black", alpha=0.8, lw=1.0,
               label="baseline = " + (f"{bl_rate*100:.1f}%" if metric == "rate"
                                      else f"{bl_logit:+.2f}"))
    ax.set_xlabel("Steering coefficient")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3, linestyle=":")
    if put_legend:
        ax.legend(loc="best", fontsize=8, ncol=2, framealpha=0.9)


def make_fig6():
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5))
    for col, (model_title, res_dir) in enumerate(MODEL_SOURCES):
        rates = load_rates(res_dir)
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


def make_fig7():
    """Family-mean sweep: CAA + critical mean + conformist mean.

    Band = min/max across the 3 family members. Degraded (condition,
    coefficient) cells are excluded from the mean and the band.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5))
    for col, (model_title, res_dir) in enumerate(MODEL_SOURCES):
        rates = load_rates(res_dir)
        degraded = load_degraded(res_dir)
        _plot_family(axes[0, col], rates, degraded, "rate",
                     f"{model_title}: sycophancy rate",
                     "Sycophancy rate (%)",
                     put_legend=(col == 0))
        _plot_family(axes[1, col], rates, degraded, "logit",
                     f"{model_title}: mean sycophancy logit",
                     r"Mean sycophancy logit  $\log p(\mathrm{syc}) - \log p(\mathrm{hon})$",
                     put_legend=False)
    fig.suptitle(
        "Family-averaged sycophancy response to steering coefficient\n"
        "CAA (single vector) + critical mean (n=3) + conformist mean (n=3); "
        "bands = min/max across members; degraded cells masked",
        fontsize=11,
    )
    plt.tight_layout()
    save(fig, str(FIG / "fig7_steering_curves_family"))
    plt.close(fig)


def main():
    print("Building fig6_steering_curves...")
    make_fig6()
    print(f"  saved fig6_steering_curves.{{pdf,png}} in {FIG}")
    print("Building fig7_steering_curves_family...")
    make_fig7()
    print(f"  saved fig7_steering_curves_family.{{pdf,png}} in {FIG}")


if __name__ == "__main__":
    main()
