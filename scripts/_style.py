"""Shared matplotlib styling. Deterministic (no RNG).

Colorblind-safe palette tested for deuteranopia and protanopia:
critical roles = blue family, conformist roles = orange family,
CAA = dark neutral gray, random null band = light neutral gray.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Nimbus Roman", "serif"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": ":",
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,       # embed TrueType so PDF text is selectable
    "ps.fonttype": 42,
})

PALETTE = {
    "caa":           "#4d4d4d",
    "skeptic":       "#08519c",
    "devils_advocate":"#2171b5",
    "judge":         "#4292c6",
    "peacekeeper":   "#d94801",
    "pacifist":      "#f16913",
    "collaborator":  "#fd8d3c",
    "random":        "#bdbdbd",
}

LABELS = {
    "caa":            "CAA (targeted)",
    "skeptic":        "Skeptic",
    "devils_advocate":"Devil's Advocate",
    "judge":          "Judge",
    "peacekeeper":    "Peacekeeper",
    "pacifist":       "Pacifist",
    "collaborator":   "Collaborator",
    "random":         "Random (n=10)",
}

CRITICAL   = ["skeptic", "devils_advocate", "judge"]
CONFORMIST = ["peacekeeper", "pacifist", "collaborator"]

# x-axis order for bar charts: CAA | critical | conformist | random
DEFAULT_ORDER = ["caa"] + CRITICAL + CONFORMIST + ["random"]


def sig_marker(n_sig, n_seeds=3):
    """Asterisks for per-seed Holm significance count."""
    if n_sig is None:
        return ""
    if n_sig == n_seeds:
        return "***"
    if n_sig == n_seeds - 1:
        return "**"
    if n_sig > 0:
        return "*"
    return ""


def save(fig, out_stem):
    """Save to {stem}.png (300 dpi) and {stem}.pdf (vector)."""
    fig.savefig(f"{out_stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{out_stem}.pdf", bbox_inches="tight")
