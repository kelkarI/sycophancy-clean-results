"""Render fig9 (per-layer mean cos(ΔH^CAA, ΔH^persona)) and fig10
(per-layer mean ‖ΔH_ℓ‖) as two-panel side-by-side figures, Gemma left
and Qwen right. Reads data/{model}_perturbation_propagation.json.

Style follows scripts/_style.py (PALETTE, LABELS, save). Vertical
dashed line at TARGET_LAYER on each panel; horizontal dotted at 0
on fig9.

Line styles encode the family the persona belongs to:
  • critical roles  (skeptic, devils_advocate, judge)        — solid
  • conformist roles (peacekeeper, pacifist, collaborator,
                      facilitator)                           — solid, lighter
  • random_0 control                                         — dashed, grey
PALETTE keys are reused: random_0 maps to PALETTE['random']
(legend label "Random (n=10)" — kept for consistency with fig1-fig7
even though this experiment uses a single random unit vector).
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
    ("Gemma 2 27B", DATA / "gemma-2-27b-it_perturbation_propagation.json"),
    ("Qwen 3 32B",  DATA / "qwen3-32b_perturbation_propagation.json"),
]


CONFORMIST_ALL = CONFORMIST + ["facilitator"]  # fig9-10 keeps facilitator


def _palette_key(condition: str) -> str:
    """Driver writes 'random_0' as the random-control name; map it to
    the global PALETTE/LABELS key 'random' for consistency with fig1-fig7."""
    if condition.startswith("random"):
        return "random"
    return condition


def _line_style(condition: str) -> dict:
    """Encode persona family in the line style on top of the PALETTE color.
    Critical: solid, lw=1.8. Conformist: solid, lw=1.3. Random: dashed, lw=1.3."""
    if condition.startswith("random"):
        return dict(linestyle="--", linewidth=1.3, alpha=0.95)
    if condition in CRITICAL:
        return dict(linestyle="-", linewidth=1.9, alpha=0.95)
    if condition in CONFORMIST_ALL:
        return dict(linestyle="-", linewidth=1.3, alpha=0.85)
    if condition == "caa":
        return dict(linestyle="-", linewidth=2.0, alpha=0.95)
    return dict(linestyle="-", linewidth=1.3, alpha=0.85)


def _legend_order(personas):
    """Critical | Conformist | Random — keeps the legend readable across
    the 8-line plot. Falls back to input order for unknowns."""
    crit  = [p for p in CRITICAL if p in personas]
    conf  = [p for p in CONFORMIST_ALL if p in personas]
    rand  = [p for p in personas if p.startswith("random")]
    rest  = [p for p in personas if p not in crit + conf + rand]
    return crit + conf + rand + rest


def _load(p: Path) -> dict:
    return json.loads(p.read_text())


def _plot_cosine_panel(ax, block: dict, model_label: str):
    layers = np.array(block["layers"])
    target = int(block["target_layer"])
    for persona in _legend_order(block["personas_compared_to_caa"]):
        cs = block["cosine"][persona]
        m = np.array(cs["mean"])
        lo = np.array(cs["ci_lo"])
        hi = np.array(cs["ci_hi"])
        key = _palette_key(persona)
        style = _line_style(persona)
        ax.plot(layers, m, label=LABELS.get(key, persona),
                color=PALETTE.get(key), **style)
        ax.fill_between(layers, lo, hi, alpha=0.15, color=PALETTE.get(key))
    ax.axhline(0.0, color="k", lw=0.5, ls=":")
    ax.axvline(target, color="k", lw=0.6, ls="--",
               label=f"TARGET={target}")
    ax.set_xlabel(r"Layer $\ell$ (post-block residual)")
    ax.set_ylabel(r"mean $\cos(\Delta H^{\mathrm{CAA}}_\ell,\,\Delta H^{\mathrm{persona}}_\ell)$")
    ax.set_title(model_label)
    ax.legend(loc="best", fontsize=7, ncol=1)


def _plot_norm_panel(ax, block: dict, model_label: str):
    layers = np.array(block["layers"])
    target = int(block["target_layer"])
    # Plot order: critical | conformist | random | caa (so caa sits on top)
    norm_order = _legend_order([c for c in block["conditions_for_norm"]
                                if c != "caa"]) + ["caa"]
    for cn in norm_order:
        if cn not in block["norm"]:
            continue
        ns = block["norm"][cn]
        m = np.array(ns["mean"])
        lo = np.array(ns["ci_lo"])
        hi = np.array(ns["ci_hi"])
        key = _palette_key(cn)
        style = _line_style(cn)
        ax.plot(layers, m, label=LABELS.get(key, cn),
                color=PALETTE.get(key), **style)
        ax.fill_between(layers, lo, hi, alpha=0.15, color=PALETTE.get(key))
    ax.axvline(target, color="k", lw=0.6, ls="--")
    ax.set_xlabel(r"Layer $\ell$ (post-block residual)")
    ax.set_ylabel(r"mean $\|\Delta H_\ell\|_2$ over (tokens, prompts)")
    ax.set_title(model_label)
    ax.set_yscale("log")  # condition magnitudes vary by 2-3 OOM downstream
    ax.legend(loc="best", fontsize=7, ncol=1)


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
