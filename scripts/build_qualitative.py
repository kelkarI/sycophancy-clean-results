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


def _render_gemma_showcase(qi: int = 2) -> None:
    """Render the John-Locke-empiricism prompt across conditions."""
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
    """Readable card layout: one prompt per model, 4 responses stacked."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch, Rectangle
    except ImportError:
        print("matplotlib not available; skipping figure")
        return
    import sys
    sys.path.insert(0, str(HERE / "scripts"))
    from _style import PALETTE, save  # noqa: E402

    caa = json.loads((OUT / "qual_check_caa.json").read_text())
    conf = json.loads((OUT / "qual_check_conformist.json").read_text())
    gemma_by = defaultdict(dict)
    for r in caa["results"] + conf["results"]:
        gemma_by[r["qi"]][r["condition"]] = r
    qw = json.loads((OUT / "qwen3-32b_over_correction.json").read_text())
    qwen_by = defaultdict(dict)
    for r in qw["per_probe_results"]:
        qwen_by[r["topic"]][r["condition"]] = r

    # Representative prompts:
    # Gemma qi=2 (John Locke): baseline "knowledge aligns with your empiricist perspective"
    # → caa "the question is complex" → skeptic "I must respectfully disagree"
    # → pacifist "profound agreement". All four tones are signature-phrase visible.
    # Qwen chemistry-false: baseline/caa hedge-correct, skeptic explicit refuse,
    # pacifist degenerates into a repetition loop (the locked-coef collapse).
    gemma_row = gemma_by[2]
    qwen_row = qwen_by["chemistry (false)"]
    gemma_prompt = gemma_row["baseline"]["prompt_ab"]
    qwen_prompt = [p for p in qw["probes"] if p["topic"] == "chemistry (false)"][0]["prompt"]

    order = ["baseline", "caa", "skeptic", "pacifist"]
    pretty = {
        "baseline":  ("Baseline (no steering)",   "—",          "#555555"),
        "caa":       ("CAA (targeted)",           "targeted",   PALETTE["caa"]),
        "skeptic":   ("Skeptic",                  "critical",   PALETTE["skeptic"]),
        "pacifist":  ("Pacifist",                 "conformist", PALETTE["pacifist"]),
    }

    def _clean(s: str) -> str:
        return "".join(c for c in s if ord(c) < 0x2E80).strip()

    import re
    _HONORIFIC = re.compile(r"\b(Mr|Mrs|Ms|Dr|Prof|St|Sr|Jr|H2O|e\.g|i\.e|vs)\.$")

    def _split_two(text: str, head_min: int = 70, head_max: int = 180, body_chars: int = 320):
        """Pull the signature opening sentence, then the next sentence(s)."""
        t = _clean(text).replace("\n\n", "  ").replace("\n", " ")
        t = " ".join(t.split())
        # scan for sentence-ending . ! ? that isn't an honorific / abbreviation
        cut = -1
        for m in re.finditer(r"[.!?](?=\s|$)", t):
            end = m.end()
            if end < head_min:
                continue
            if end > head_max:
                break
            prefix = t[:end]
            if _HONORIFIC.search(prefix):
                continue
            cut = end
            break
        if cut == -1:
            head = t[:head_max].rsplit(" ", 1)[0] + " …"
            tail_start = len(head)
        else:
            head = t[:cut].strip()
            tail_start = cut
        tail = t[tail_start:].strip()
        if len(tail) > body_chars:
            tail = tail[:body_chars].rsplit(" ", 1)[0] + " …"
        return head, tail

    import textwrap

    fig = plt.figure(figsize=(16, 15))
    gs = fig.add_gridspec(
        nrows=6, ncols=2,
        height_ratios=[0.6, 1.0, 1.0, 1.0, 1.0, 0.08],
        hspace=0.22, wspace=0.10,
        left=0.04, right=0.98, top=0.94, bottom=0.02,
    )

    def _prompt_panel(ax, model_name, prompt_text):
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ("top","right","bottom","left"):
            ax.spines[s].set_visible(False)
        ax.grid(False)
        ax.add_patch(Rectangle((0,0),1,1, facecolor="#f5f5f5",
                               edgecolor="#cccccc", linewidth=0.8))
        ax.text(0.015, 0.92, model_name, fontsize=13, fontweight="bold",
                va="top", color="#111")
        ax.text(0.015, 0.70, "User prompt", fontsize=9, va="top",
                color="#888", fontstyle="italic")
        body = _clean(prompt_text)
        # collapse whitespace and truncate for display
        body = " ".join(body.split())
        if len(body) > 420:
            body = body[:420].rsplit(" ", 1)[0] + " …"
        wrapped = textwrap.fill(body, width=95)
        ax.text(0.015, 0.57, wrapped, fontsize=9.5, va="top", color="#222",
                family="serif")

    def _response_panel(ax, cond, r, metric_kind):
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ("top","right","bottom","left"):
            ax.spines[s].set_visible(False)
        ax.grid(False)
        label, family, color = pretty[cond]
        ax.add_patch(Rectangle((0,0),1,1, facecolor="white",
                               edgecolor="#dddddd", linewidth=0.8))
        ax.add_patch(Rectangle((0,0), 0.012, 1, facecolor=color, linewidth=0))
        ax.add_patch(Rectangle((0.012,0.82), 0.988, 0.18,
                               facecolor=color, linewidth=0, alpha=0.92))
        ax.text(0.03, 0.93, label, fontsize=12, fontweight="bold",
                color="white", va="center")
        ax.text(0.03, 0.855, family, fontsize=8.5, color="white",
                va="center", fontstyle="italic", alpha=0.95)
        if metric_kind == "syc_logit":
            badge = f"syc_logit {r['syc_logit']:+.2f}"
            badge2 = f"coef {r['coefficient']:+.0f}   A/B = {r['chosen']}"
        else:
            badge = f"{r['category']}"
            badge2 = f"coef {r['coefficient']:+.0f}   correct = {r['correct']}"
        if cond == "pacifist" and metric_kind == "category":
            badge += "  · DEGRADED"
        ax.text(0.985, 0.93, badge, fontsize=10.5, fontweight="bold",
                color="white", ha="right", va="center",
                family="monospace")
        ax.text(0.985, 0.855, badge2, fontsize=8.5,
                color="white", ha="right", va="center",
                fontstyle="italic", alpha=0.95)
        head, tail = _split_two(r["response"])
        head_wrapped = textwrap.fill(head, width=84)
        tail_wrapped = textwrap.fill(tail, width=96)
        n_head_lines = head_wrapped.count("\n") + 1
        ax.text(0.025, 0.73, head_wrapped, fontsize=11, fontweight="bold",
                color=color, va="top", family="serif", linespacing=1.25)
        tail_y = 0.73 - 0.075 * n_head_lines - 0.02
        ax.text(0.025, tail_y, tail_wrapped, fontsize=9.5,
                color="#333", va="top", family="serif", linespacing=1.3)

    _prompt_panel(fig.add_subplot(gs[0, 0]),
                  "Gemma 2 27B  ·  John Locke / empiricism", gemma_prompt)
    _prompt_panel(fig.add_subplot(gs[0, 1]),
                  "Qwen 3 32B  ·  Chemistry professor (false claim)", qwen_prompt)
    for i, cond in enumerate(order):
        _response_panel(fig.add_subplot(gs[i + 1, 0]),
                        cond, gemma_row[cond], "syc_logit")
        _response_panel(fig.add_subplot(gs[i + 1, 1]),
                        cond, qwen_row[cond], "category")

    fig.suptitle(
        "How steering changes tone — one representative prompt per model, "
        "each condition at its tune-locked coefficient",
        fontsize=14, fontweight="bold", y=0.985,
    )
    footer = fig.add_subplot(gs[5, :])
    footer.axis("off")
    footer.text(
        0.5, 0.5,
        "Bold = opening sentence (tone signature). "
        "Gemma metric = syc_logit on the A/B continuation (lower is less sycophantic). "
        "Qwen metric = auto-category from sycophancy-qwen pipeline. "
        "Full text in qualitative/{gemma,qwen}_showcase.md.",
        ha="center", va="center", fontsize=9, color="#666",
        fontstyle="italic",
    )
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
