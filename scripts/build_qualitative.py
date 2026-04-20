"""Rebuild qualitative/ from the source repos.

Pulls three sets of stored decoded responses:

1. Gemma philosophy-prompt free-form responses
   (`sycophancy-final/experiment-main/results/qual_check_{caa,conformist}.json`)
   — 5 prompts × {baseline, caa @ -2000, assistant_axis @ +2000,
   skeptic @ +2000, peacekeeper @ +2000, pacifist @ +2000,
   collaborator @ +500, facilitator @ +2000}.
2. Gemma over-correction probes
   (`sycophancy-final/experiment-main/results/over_correction_eval_test.json`)
   — 16 factual/attitudinal probes across 14 conditions.
3. Qwen over-correction probes
   (`sycophancy-qwen/results/over_correction_eval_test.json`)
   — same schema, 16 probes × 16 conditions. Qwen has no
   philosophy-prompt qual file.

We keep only the conditions listed in the clean scope (critical +
conformist + caa + baseline + random_0) and write trimmed JSON plus
human-readable markdown showcases for one representative probe per
model.

CPU-only, deterministic. Source paths are hard-coded to the canonical
checkout layout — adjust if the source repos live elsewhere.
"""

from __future__ import annotations

import json
import textwrap
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
OUT = HERE / "qualitative"
FIGOUT = HERE / "figures"
OUT.mkdir(exist_ok=True)
FIGOUT.mkdir(exist_ok=True)

GEMMA_SRC = Path("/home/user/sycophancy-final/experiment-main/results")
QWEN_SRC = Path("/home/user/sycophancy-qwen/results")

KEPT = {
    "baseline",
    "skeptic",
    "devils_advocate",
    "judge",
    "peacekeeper",
    "pacifist",
    "collaborator",
    "caa",
    "random_0",
}


def _slim_over_correction(src: Path, model: str) -> dict:
    d = json.loads(src.read_text())
    kept = [r for r in d["per_probe_results"] if r["condition"] in KEPT]
    return {
        "model": model,
        "split": d.get("split"),
        "kept_conditions": sorted({r["condition"] for r in kept}),
        "coefficients_by_condition": {
            c: v for c, v in d.get("coefficients_by_condition", {}).items() if c in KEPT
        },
        "probes": d["probes"],
        "per_probe_results": kept,
    }


def _copy_gemma_qual() -> None:
    for name in ("qual_check_caa.json", "qual_check_conformist.json"):
        src = GEMMA_SRC / name
        if src.exists():
            (OUT / name).write_text(src.read_text())


def _render_gemma_showcase(qi: int = 4) -> None:
    """Render the John-Smith-classical-logic prompt across conditions."""
    caa = json.loads((OUT / "qual_check_caa.json").read_text())
    conf = json.loads((OUT / "qual_check_conformist.json").read_text())
    by = defaultdict(dict)
    for r in caa["results"] + conf["results"]:
        by[r["qi"]][r["condition"]] = r
    row = by[qi]
    prompt = row["baseline"]["prompt_ab"]
    order = ["baseline", "caa", "skeptic", "collaborator", "peacekeeper", "pacifist"]
    md = [
        "# Gemma 2 27B — qualitative tone showcase",
        "",
        f"Single held-out prompt (qi={qi}) from `qual_check_*.json` decoded at each",
        "condition's tune-locked coefficient. `syc_logit` is the logit gap",
        "`logp(syc_token) - logp(hon_token)` scored on the A/B continuation —",
        "lower means less sycophantic. The free-form text above is the actual",
        "model continuation under steering.",
        "",
        "## Prompt",
        "```",
        prompt.strip(),
        "```",
        "",
    ]
    for cond in order:
        r = row.get(cond)
        if r is None:
            continue
        family = {
            "baseline": "baseline (no steering)",
            "caa": "caa (targeted)",
            "skeptic": "critical",
            "collaborator": "conformist",
            "peacekeeper": "conformist",
            "pacifist": "conformist",
        }[cond]
        md += [
            f"## {cond} — {family}, coef={r['coefficient']:+.0f}",
            "",
            f"`syc_logit` = **{r['syc_logit']:+.2f}**  "
            f"(A/B pick = {r['chosen']}, sycophantic = {r['chose_sycophantic']})",
            "",
            "> " + r["response"].strip().replace("\n", "\n> "),
            "",
        ]
    (OUT / "gemma_showcase.md").write_text("\n".join(md))


def _render_qwen_showcase() -> None:
    """Render the arithmetic and chemistry(false) probes across conditions."""
    qw = json.loads((OUT / "qwen3-32b_over_correction.json").read_text())
    by = defaultdict(dict)
    for r in qw["per_probe_results"]:
        by[r["topic"]][r["condition"]] = r
    probes = {p["topic"]: p for p in qw["probes"]}
    order = ["baseline", "caa", "skeptic", "devils_advocate", "judge",
             "peacekeeper", "collaborator", "pacifist"]
    md = ["# Qwen 3 32B — qualitative tone showcase", "",
          "Two held-out probes decoded under each kept condition at its",
          "tune-locked coefficient. `category` is the pipeline's own",
          "auto-label: `AGREE_CORRECT` / `AGREE_INCORRECT` / `REFUSE` / `HEDGE`.",
          "Pacifist at coef +500 is the one cell flagged as degraded — its",
          "forward pass collapses into a repetitive loop, which is what a",
          "model-collapse artefact looks like in text.",
          ""]
    for topic in ("arithmetic", "chemistry (false)"):
        probe = probes[topic]
        md += [f"## Probe: {topic} (true_claim = {probe['true_claim']})", "",
               "```", probe["prompt"].strip(), "```", ""]
        for cond in order:
            r = by[topic].get(cond)
            if r is None:
                continue
            family = {
                "baseline": "baseline",
                "caa": "caa (targeted)",
                "skeptic": "critical", "devils_advocate": "critical", "judge": "critical",
                "peacekeeper": "conformist", "collaborator": "conformist",
                "pacifist": "conformist",
            }[cond]
            degraded = ""
            if cond == "pacifist":
                degraded = "  **[degraded at this coef]**"
            md += [
                f"### {cond} — {family}, coef={r['coefficient']:+.0f}{degraded}",
                "",
                f"category = **{r['category']}**, correct = {r['correct']}",
                "",
                "> " + r["response"].strip().replace("\n", "\n> "),
                "",
            ]
    (OUT / "qwen_showcase.md").write_text("\n".join(md))


def _make_tone_figure() -> None:
    """2x4 card grid of response snippets across conditions on both models."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch
    except ImportError:
        print("matplotlib not available; skipping figure")
        return
    import sys
    sys.path.insert(0, str(HERE / "scripts"))
    from _style import PALETTE, LABELS, save  # noqa: E402

    caa = json.loads((OUT / "qual_check_caa.json").read_text())
    conf = json.loads((OUT / "qual_check_conformist.json").read_text())
    gemma_by = defaultdict(dict)
    for r in caa["results"] + conf["results"]:
        gemma_by[r["qi"]][r["condition"]] = r
    qw = json.loads((OUT / "qwen3-32b_over_correction.json").read_text())
    qwen_by = defaultdict(dict)
    for r in qw["per_probe_results"]:
        qwen_by[r["topic"]][r["condition"]] = r

    gemma_row = gemma_by[4]
    qwen_row = qwen_by["arithmetic"]
    cols = ["baseline", "caa", "skeptic", "pacifist"]
    col_titles = ["Baseline (no steering)", "CAA (targeted)",
                  "Skeptic (critical)", "Pacifist (conformist)"]
    palette = {
        "baseline": "#666666",
        "caa": PALETTE["caa"],
        "skeptic": PALETTE["skeptic"],
        "pacifist": PALETTE["pacifist"],
    }

    def _snippet(text: str, n_chars: int = 340) -> str:
        t = text.strip().replace("\n\n", " ¶ ").replace("\n", " ")
        # strip characters outside the BMP range that the serif font can't render
        t = "".join(c for c in t if ord(c) < 0x2E80)
        if len(t) > n_chars:
            t = t[: n_chars].rsplit(" ", 1)[0] + " …"
        return textwrap.fill(t, width=48)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    model_rows = [("Gemma 2 27B", "Prof. John Smith: classical vs non-classical logic",
                   gemma_row, "syc_logit"),
                  ("Qwen 3 32B", "Math teacher: '2 + 2 = 4, do you agree?'",
                   qwen_row, "category")]
    for row_i, (model_title, prompt_title, row, metric) in enumerate(model_rows):
        for col_i, cond in enumerate(cols):
            ax = axes[row_i, col_i]
            ax.set_xticks([]); ax.set_yticks([])
            for side in ("top", "right", "bottom", "left"):
                ax.spines[side].set_visible(False)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.grid(False)
            r = row.get(cond)
            if r is None:
                ax.text(0.5, 0.5, "(not recorded)", ha="center", va="center",
                        color="#999", fontsize=10)
                continue
            color = palette[cond]
            bar = FancyBboxPatch((0, 0.93), 1, 0.07,
                                 boxstyle="round,pad=0,rounding_size=0.01",
                                 linewidth=0, facecolor=color)
            ax.add_patch(bar)
            head = f"{col_titles[col_i]}   coef={r['coefficient']:+.0f}"
            ax.text(0.02, 0.965, head, color="white", fontsize=10,
                    fontweight="bold", va="center")
            if metric == "syc_logit":
                sub = f"syc_logit = {r['syc_logit']:+.2f}   (A/B = {r['chosen']})"
            else:
                sub = f"category = {r['category']}   correct = {r['correct']}"
            if cond == "pacifist" and row_i == 1:
                sub += "    [DEGRADED]"
            ax.text(0.02, 0.88, sub, color=color, fontsize=9,
                    fontweight="bold", va="top")
            ax.text(0.02, 0.82, _snippet(r["response"]),
                    color="#222", fontsize=8.3, va="top", wrap=True,
                    family="serif")
        axes[row_i, 0].text(-0.05, 0.5,
                            f"{model_title}\n{prompt_title}",
                            transform=axes[row_i, 0].transAxes,
                            rotation=90, va="center", ha="center",
                            fontsize=10, fontstyle="italic", color="#444")

    fig.suptitle(
        "Tone under steering at tune-locked coefficients — one held-out prompt per model",
        fontsize=12, y=0.995,
    )
    fig.text(0.5, 0.02,
             "Snippets are the first ~340 characters of the decoded response. "
             "Full text in qualitative/{gemma,qwen}_showcase.md.",
             ha="center", color="#666", fontsize=8.5, fontstyle="italic")
    fig.tight_layout(rect=(0.02, 0.04, 1, 0.98))
    save(fig, str(FIGOUT / "fig5_tone_comparison"))
    plt.close(fig)


def main() -> None:
    _copy_gemma_qual()
    if GEMMA_SRC.exists():
        gemma = _slim_over_correction(
            GEMMA_SRC / "over_correction_eval_test.json", "gemma-2-27b-it"
        )
        (OUT / "gemma-2-27b-it_over_correction.json").write_text(
            json.dumps(gemma, ensure_ascii=False, indent=2)
        )
    if QWEN_SRC.exists():
        qwen = _slim_over_correction(
            QWEN_SRC / "over_correction_eval_test.json", "qwen3-32b"
        )
        (OUT / "qwen3-32b_over_correction.json").write_text(
            json.dumps(qwen, ensure_ascii=False, indent=2)
        )
    if (OUT / "qual_check_caa.json").exists():
        _render_gemma_showcase()
    if (OUT / "qwen3-32b_over_correction.json").exists():
        _render_qwen_showcase()
    _make_tone_figure()
    print("wrote", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
