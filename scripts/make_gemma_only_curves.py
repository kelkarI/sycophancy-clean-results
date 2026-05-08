"""Gemma-only family-mean steering curves -- single-panel (rate).

Variant of `make_steering_curves.py::make_fig7` restricted to Gemma and
to the sycophancy-rate metric. Produces one wide axis instead of the
2x2 grid, intended for a single-column paper figure where the logit
panel has been dropped and the rate panel should fill the row.

Output: `fig3_gemma_steering_curves.{pdf,png}` in `figures/`.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _style import save
from make_steering_curves import (
    GEMMA_RES,
    _plot_family,
    load_degraded,
    load_rates,
)

ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def make():
    rates = load_rates(GEMMA_RES)
    degraded = load_degraded(GEMMA_RES)
    fig, ax = plt.subplots(1, 1, figsize=(7.0, 4.6))
    _plot_family(
        ax, rates, degraded, "rate",
        "Gemma Sycophancy Rate",
        "Sycophancy rate (%)",
        put_legend=True,
    )
    fig.suptitle("Gemma Response to Steering Coefficient Curves", fontsize=12)
    plt.tight_layout()
    save(fig, str(FIG / "fig3_gemma_steering_curves"))
    plt.close(fig)


if __name__ == "__main__":
    make()
    print(f"  saved fig3_gemma_steering_curves.{{pdf,png}} in {FIG}")
