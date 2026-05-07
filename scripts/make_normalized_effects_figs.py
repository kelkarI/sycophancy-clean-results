"""Render fig11 — behavioral efficiency = |∆logit| / ε per condition,
two-panel side-by-side (Gemma left, Qwen right).

The whole point of fig11 is to ask "do persona vectors exceed random
at matched perturbation energy". The random_0 bar is therefore
rendered with a horizontal dashed reference line spanning the whole
panel, so a reader sees at a glance which conditions clear the
random-baseline bar and by how much.

Reads data/{model}_normalized_effects.json. PALETTE / LABELS / save
from scripts/_style.py.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _style import PALETTE, LABELS, CRITICAL, CONFORMIST, save


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FIG  = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

MODEL_FILES = [
    ("Gemma 2 27B", DATA / "gemma-2-27b-it_normalized_effects.json"),
    ("Qwen 3 32B",  DATA / "qwen3-32b_normalized_effects.json"),
]

CONFORMIST_ALL = CONFORMIST + ["facilitator"]
ORDER = ["caa"] + CRITICAL + CONFORMIST_ALL + ["random_0"]


def _palette_key(c):
    if c.startswith("random"):
        return "random"
    return c


def _hatch_for(c):
    """Random_0 gets diagonal hatching to visually flag it as the null."""
    return "//" if c.startswith("random") else None


def _plot_panel(ax, block: dict, model_label: str):
    conds = block["conditions"]
    plotted = [c for c in ORDER if c in conds and conds[c].get("behavioral_efficiency") is not None]

    xs = np.arange(len(plotted))
    ys = [conds[c]["behavioral_efficiency"] for c in plotted]
    colors = [PALETTE.get(_palette_key(c), "#888888") for c in plotted]

    bars = ax.bar(xs, ys, color=colors, edgecolor="black", linewidth=0.5)
    for b, c in zip(bars, plotted):
        h = _hatch_for(c)
        if h:
            b.set_hatch(h)
            b.set_edgecolor("#222222")

    # Random reference line
    if "random_0" in conds and conds["random_0"].get("behavioral_efficiency") is not None:
        rand_y = conds["random_0"]["behavioral_efficiency"]
        ax.axhline(rand_y, color=PALETTE.get("random"), linestyle="--",
                   linewidth=1.2, alpha=0.85, zorder=0,
                   label=f"random_0 baseline = {rand_y:.2f}")

    # Annotate bars with ε and Δlogit (compact)
    for x, c in zip(xs, plotted):
        eps = conds[c]["perturbation_energy"]
        dl = conds[c]["delta_logit_mean"]
        eff = conds[c]["behavioral_efficiency"]
        ax.text(x, eff + max(ys) * 0.015,
                f"ε={eps:.2f}\nΔ={dl:+.2f}",
                ha="center", va="bottom", fontsize=6.5)

    ax.set_xticks(xs)
    ax.set_xticklabels([LABELS.get(_palette_key(c), c) for c in plotted],
                       rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(r"behavioral efficiency $|\Delta\mathrm{logit}|\,/\,\epsilon$")
    title = f"{model_label}  (TARGET={block['target_layer']}, " \
            f"‖h̄‖={block['h_baseline_norm_at_target_layer']:.0f})"
    ax.set_title(title, fontsize=10)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(bottom=0, top=max(ys) * 1.20)


def main():
    blocks = [(label, json.loads(p.read_text())) for label, p in MODEL_FILES]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, (label, b) in zip(axes, blocks):
        _plot_panel(ax, b, label)
    fig.suptitle(
        r"Sycophancy reduction per unit fractional perturbation: "
        r"$|\Delta\mathrm{logit}|\,/\,\epsilon$, with $\epsilon = |\alpha|/\|\bar{h}^{\mathrm{baseline}}_{\mathrm{TARGET}}\|$",
        y=1.04, fontsize=11,
    )
    fig.tight_layout()
    save(fig, str(FIG / "fig11_normalized_effects"))
    plt.close(fig)
    print(f"  wrote {FIG / 'fig11_normalized_effects.png'}")
    print(f"  wrote {FIG / 'fig11_normalized_effects.pdf'}")


if __name__ == "__main__":
    main()
