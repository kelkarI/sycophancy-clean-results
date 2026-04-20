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
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
OUT = HERE / "qualitative"
OUT.mkdir(exist_ok=True)

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
    print("wrote", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
