"""Render fig9 (per-layer mean cos(ΔH^CAA, ΔH^persona)) and fig10
(per-layer mean ‖ΔH_ℓ‖) as two-panel side-by-side figures, Gemma left
and Qwen right. Reads data/{model}_perturbation_propagation.json.

Style follows scripts/_style.py (PALETTE, LABELS, save). Vertical
dashed line at TARGET_LAYER on each panel; horizontal dotted at 0
on fig9. Random_0 condition maps to PALETTE['random'] /
LABELS['random'] (so the legend reads "Random (n=10)" even though
this experiment uses a single random unit vector — the palette key
stays consistent with fig1-fig7).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _style import PALETTE, LABELS, save


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FIG  = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

MODEL_FILES = [
    ("Gemma 2 27B", DATA / "gemma-2-27b-it_perturbation_propagation.json"),
    ("Qwen 3 32B",  DATA / "qwen3-32b_perturbation_propagation.json"),
]


def _palette_key(condition: str) -> str:
    """Driver writes 'random_0' as the random-control name; map it to
    the global PALETTE/LABELS key 'random' for consistency with fig1-fig7."""
    if condition.startswith("random"):
        return "random"
    return condition


def _load(p: Path) -> dict:
    return json.loads(p.read_text())


def _plot_cosine_panel(ax, block: dict, model_label: str):
    layers = np.array(block["layers"])
    target = int(block["target_layer"])
    for persona in block["personas_compared_to_caa"]:
        cs = block["cosine"][persona]
        m = np.array(cs["mean"])
        lo = np.array(cs["ci_lo"])
        hi = np.array(cs["ci_hi"])
        key = _palette_key(persona)
        ax.plot(layers, m, label=LABELS.get(key, persona),
                color=PALETTE.get(key))
        ax.fill_between(layers, lo, hi, alpha=0.2, color=PALETTE.get(key))
    ax.axhline(0.0, color="k", lw=0.5, ls=":")
    ax.axvline(target, color="k", lw=0.6, ls="--",
               label=f"TARGET={target}")
    ax.set_xlabel(r"Layer $\ell$ (post-block residual)")
    ax.set_ylabel(r"mean $\cos(\Delta H^{\mathrm{CAA}}_\ell,\,\Delta H^{\mathrm{persona}}_\ell)$")
    ax.set_title(model_label)
    ax.legend(loc="best", fontsize=8)


def _plot_norm_panel(ax, block: dict, model_label: str):
    layers = np.array(block["layers"])
    target = int(block["target_layer"])
    for cn in block["conditions_for_norm"]:
        if cn not in block["norm"]:
            continue
        ns = block["norm"][cn]
        m = np.array(ns["mean"])
        lo = np.array(ns["ci_lo"])
        hi = np.array(ns["ci_hi"])
        key = _palette_key(cn)
        ax.plot(layers, m, label=LABELS.get(key, cn),
                color=PALETTE.get(key))
        ax.fill_between(layers, lo, hi, alpha=0.2, color=PALETTE.get(key))
    ax.axvline(target, color="k", lw=0.6, ls="--")
    ax.set_xlabel(r"Layer $\ell$ (post-block residual)")
    ax.set_ylabel(r"mean $\|\Delta H_\ell\|_2$ over (tokens, prompts)")
    ax.set_title(model_label)
    ax.set_yscale("log")  # condition magnitudes vary by 2-3 OOM downstream
    ax.legend(loc="best", fontsize=8)


def main():
    blocks = [(label, _load(p)) for label, p in MODEL_FILES]

    # ----- fig9: per-layer cosine, two panels side-by-side -----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), sharey=True)
    for ax, (label, b) in zip(axes, blocks):
        _plot_cosine_panel(ax, b, label)
    fig.suptitle(
        r"Per-layer perturbation cosine: $\cos(\Delta H^{\mathrm{CAA}}_\ell, "
        r"\Delta H^{\mathrm{persona}}_\ell)$",
        y=1.02,
    )
    fig.tight_layout()
    save(fig, str(FIG / "fig9_perturbation_cosine"))
    plt.close(fig)

    # ----- fig10: per-layer norm, two panels side-by-side -----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    for ax, (label, b) in zip(axes, blocks):
        _plot_norm_panel(ax, b, label)
    fig.suptitle(
        r"Per-layer perturbation magnitude: $\|\Delta H_\ell\|_2$",
        y=1.02,
    )
    fig.tight_layout()
    save(fig, str(FIG / "fig10_perturbation_norm"))
    plt.close(fig)

    print(f"  wrote {FIG / 'fig9_perturbation_cosine.png'}")
    print(f"  wrote {FIG / 'fig9_perturbation_cosine.pdf'}")
    print(f"  wrote {FIG / 'fig10_perturbation_norm.png'}")
    print(f"  wrote {FIG / 'fig10_perturbation_norm.pdf'}")


if __name__ == "__main__":
    main()
