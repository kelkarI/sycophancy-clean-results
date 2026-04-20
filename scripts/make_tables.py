"""Emit the paper-ready tables.

Core tables:
  results/main_table.csv              one row per (model, condition)
  results/conformist_vs_critical.csv  family-level summary

Degradation-aware variants (cells where the source pipeline's
degradation flag is True at the tune-locked coefficient are masked
out — those cells are model-collapse artefacts, not sycophancy
reductions):
  results/main_table_filtered.csv             dropped degraded rows
  results/conformist_vs_critical_filtered.csv family mean excluding
                                              degraded members

Both core and filtered tables have .md siblings for easy pasting.
"""
import csv
import json
from pathlib import Path

import numpy as np

from _style import DEFAULT_ORDER, LABELS, CRITICAL, CONFORMIST

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT  = ROOT / "results"
OUT.mkdir(parents=True, exist_ok=True)

MODELS = {
    "gemma-2-27b-it": DATA / "gemma-2-27b-it_clean.json",
    "qwen3-32b":      DATA / "qwen3-32b_clean.json",
}


def load(p):
    return json.loads(Path(p).read_text())


def main_table():
    fieldnames = [
        "model", "family", "condition", "display_name",
        "locked_best_coef",
        "delta_logit_mean", "delta_logit_std",
        "delta_logit_ci95_lo", "delta_logit_ci95_hi",
        "delta_rate_mean_pp", "delta_rate_std_pp",
        "delta_rate_ci95_lo_pp", "delta_rate_ci95_hi_pp",
        "per_seed_delta_logit",
        "per_seed_wilcoxon_p_holm_adj",
        "n_significant_after_holm_seeds",
        "n_seeds",
        "degraded_any_seed", "degraded_all_seeds",
    ]
    rows = []
    for model, p in MODELS.items():
        block = load(p)
        for c in DEFAULT_ORDER:
            cd = block["conditions"][c]
            if c == "random":
                per_seed_d = [s["mean_delta_logit"] for s in cd["per_seed"]]
                per_seed_p = []
                n_sig = None
                coef = ""
            else:
                per_seed_d = [s["delta_logit"] for s in cd["per_seed"]]
                per_seed_p = [s["wilcoxon_p_adj"] for s in cd["per_seed"]]
                n_sig = cd.get("n_significant_after_holm_per_seed")
                coef = cd.get("locked_best_coef_aggregate", "")
            rows.append({
                "model": model,
                "family": cd["family"],
                "condition": c,
                "display_name": LABELS[c],
                "locked_best_coef": coef,
                "delta_logit_mean": round(cd["delta_logit_mean"], 4),
                "delta_logit_std":  round(cd["delta_logit_std"], 4),
                "delta_logit_ci95_lo": round(cd["delta_logit_ci95_lo"], 4),
                "delta_logit_ci95_hi": round(cd["delta_logit_ci95_hi"], 4),
                "delta_rate_mean_pp": round(cd["delta_rate_mean_pp"], 3),
                "delta_rate_std_pp":  round(cd["delta_rate_std_pp"], 3),
                "delta_rate_ci95_lo_pp": round(cd["delta_rate_ci95_lo_pp"], 3),
                "delta_rate_ci95_hi_pp": round(cd["delta_rate_ci95_hi_pp"], 3),
                "per_seed_delta_logit": ";".join(f"{v:+.4f}" for v in per_seed_d),
                "per_seed_wilcoxon_p_holm_adj": ";".join(
                    (f"{p:.3e}" if p is not None else "") for p in per_seed_p
                ),
                "n_significant_after_holm_seeds": n_sig if n_sig is not None else "",
                "n_seeds": cd["n_seeds"],
                "degraded_any_seed": bool(cd.get("degraded_any_seed", False)),
                "degraded_all_seeds": bool(cd.get("degraded_all_seeds", False)),
            })
    csv_path = OUT / "main_table.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {csv_path}  ({len(rows)} rows)")

    # Filtered CSV (drop rows degraded at the locked coef)
    filt_rows = [r for r in rows if not r["degraded_any_seed"]]
    csv_f = OUT / "main_table_filtered.csv"
    with csv_f.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(filt_rows)
    print(f"wrote {csv_f}  ({len(filt_rows)} rows, "
          f"{len(rows)-len(filt_rows)} dropped)")

    def fmt_sig(nsig, nseeds):
        if nsig == "" or nsig is None:
            return ""
        if nsig == nseeds: return "***"
        if nsig == nseeds - 1: return "**"
        if nsig >= 1: return "*"
        return "ns"

    def render_md(rows_in, title, blurb, path):
        lines = [
            f"# {title}",
            "",
            blurb,
            "",
            "| Model | Family | Condition | Best coef | Δlogit (mean ± std) "
            "[95% CI] | Δrate (pp) | Holm-sig seeds | Degraded |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for r in rows_in:
            dl = (f"{r['delta_logit_mean']:+.3f} ± {r['delta_logit_std']:.3f} "
                  f"[{r['delta_logit_ci95_lo']:+.3f}, "
                  f"{r['delta_logit_ci95_hi']:+.3f}]")
            dr = (f"{r['delta_rate_mean_pp']:+.2f} "
                  f"[{r['delta_rate_ci95_lo_pp']:+.2f}, "
                  f"{r['delta_rate_ci95_hi_pp']:+.2f}]")
            sig = fmt_sig(r['n_significant_after_holm_seeds'], r['n_seeds'])
            n_over = (f"{r['n_significant_after_holm_seeds']}/3 {sig}"
                      if r['n_significant_after_holm_seeds'] != "" else "—")
            coef = r['locked_best_coef'] if r['locked_best_coef'] != "" else "—"
            deg = ("yes" if r["degraded_all_seeds"]
                   else ("partial" if r["degraded_any_seed"] else "no"))
            name = r['display_name']
            if r["degraded_any_seed"]:
                name = name + " †"
            lines.append(
                f"| {r['model']} | {r['family']} | {name} | "
                f"{coef} | {dl} | {dr} | {n_over} | {deg} |"
            )
        if any(r["degraded_any_seed"] for r in rows_in):
            lines += [
                "",
                "† Degraded at the tune-locked coefficient in at least one "
                "test seed (see `results/seed_*/degradation_flags_test.json` "
                "in the source pipeline). At a degraded coefficient the "
                "steered model collapses toward uniform logits / binary "
                "rate = 0.5, so the reported Δ is a collapse artefact "
                "rather than a sycophancy reduction.",
            ]
        path.write_text("\n".join(lines) + "\n")
        print(f"wrote {path}")

    blurb_core = (
        "Negative Δ = steering reduces sycophancy. Error bars are 95% CIs "
        "across 3 test seeds (t-distribution, df=2). Holm correction is "
        "applied within each seed's 14-condition primary family from the "
        "source pipeline; columns show how many of the 3 seeds crossed "
        "α=0.05 after correction. *** = 3/3, ** = 2/3, * = 1/3, ns = 0/3. "
        "The 'Degraded' column marks rows whose locked coefficient is "
        "flagged as model-collapse by the source pipeline (see †)."
    )
    blurb_filt = (
        "Same as `main_table.md` but with any row whose locked coefficient "
        "is flagged as degraded in at least one test seed removed, so the "
        "table reflects only conditions where the steered forward pass is "
        "behaving normally. The dropped row(s) appear in the unfiltered "
        "`main_table.md` with a † marker."
    )
    render_md(rows,      "Main table — condition × model",
              blurb_core, OUT / "main_table.md")
    render_md(filt_rows, "Main table (degradation-filtered)",
              blurb_filt, OUT / "main_table_filtered.md")


def conformist_vs_critical_table():
    """Family-level summary: mean (and min/max) Δlogit within each family.

    Two passes:
      core     — all kept members
      filtered — members whose locked coef is flagged degraded in any
                 test seed are excluded from the family mean/range
    """
    fieldnames = [
        "model", "family", "n_conditions", "n_conditions_used", "n_seeds",
        "family_mean_delta_logit", "family_mean_delta_rate_pp",
        "family_min_delta_logit", "family_max_delta_logit",
        "members_holm_sig_3of3",
        "degraded_members_excluded",
    ]

    def build_rows(drop_degraded):
        rows = []
        for model, p in MODELS.items():
            block = load(p)
            conds = block["conditions"]
            for fam, members in [("critical",   CRITICAL),
                                 ("caa",        ["caa"]),
                                 ("conformist", CONFORMIST),
                                 ("null_control", ["random"])]:
                degraded_members = [
                    c for c in members
                    if conds[c].get("degraded_any_seed")
                ]
                used = [c for c in members if not (
                    drop_degraded and c in degraded_members)]
                if not used:
                    continue
                logits = [conds[c]["delta_logit_mean"] for c in used]
                rates  = [conds[c]["delta_rate_mean_pp"] for c in used]
                sig3 = sum(
                    1 for c in used
                    if c != "random"
                    and conds[c].get("n_significant_after_holm_per_seed") == 3
                )
                # In the core table we surface which members ARE degraded
                # (kept but flagged). In the filtered table we surface
                # which members were actually dropped.
                if drop_degraded:
                    marker = [c for c in members if c not in used]
                else:
                    marker = degraded_members
                rows.append({
                    "model": model,
                    "family": fam,
                    "n_conditions": len(members),
                    "n_conditions_used": len(used),
                    "n_seeds": 3,
                    "family_mean_delta_logit": round(float(np.mean(logits)), 4),
                    "family_mean_delta_rate_pp": round(float(np.mean(rates)), 3),
                    "family_min_delta_logit": round(float(np.min(logits)), 4),
                    "family_max_delta_logit": round(float(np.max(logits)), 4),
                    "members_holm_sig_3of3": sig3,
                    "degraded_members_excluded": ";".join(marker),
                })
        return rows

    core_rows = build_rows(drop_degraded=False)
    filt_rows = build_rows(drop_degraded=True)

    def render(rows_in, csv_name, md_name, title, blurb):
        csv_path = OUT / csv_name
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows_in)
        print(f"wrote {csv_path}  ({len(rows_in)} rows)")

        md_path = OUT / md_name
        lines = [
            f"# {title}",
            "",
            blurb,
            "",
            "| Model | Family | n cond | n used | mean Δlogit | range "
            "Δlogit | mean Δrate (pp) | Holm-sig 3/3 count | Degraded "
            "excluded |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for r in rows_in:
            excl = r["degraded_members_excluded"] or "—"
            lines.append(
                f"| {r['model']} | {r['family']} | {r['n_conditions']} "
                f"| {r['n_conditions_used']} "
                f"| {r['family_mean_delta_logit']:+.3f} "
                f"| [{r['family_min_delta_logit']:+.3f}, "
                f"{r['family_max_delta_logit']:+.3f}] "
                f"| {r['family_mean_delta_rate_pp']:+.2f} "
                f"| {r['members_holm_sig_3of3']} | {excl} |"
            )
        md_path.write_text("\n".join(lines) + "\n")
        print(f"wrote {md_path}")

    render(core_rows,
           "conformist_vs_critical.csv",
           "conformist_vs_critical.md",
           "Family-level summary — critical vs conformist roles",
           "Aggregated across the kept members of each family at their own "
           "tune-locked coefficients. `members_holm_sig_3of3` counts how "
           "many of the family's members reached Holm significance in all "
           "3 test seeds. The `Degraded excluded` column lists any "
           "members flagged as degraded (collapsed forward pass at the "
           "locked coef); those members are retained in this core table "
           "but dropped in the `_filtered` variant.")
    render(filt_rows,
           "conformist_vs_critical_filtered.csv",
           "conformist_vs_critical_filtered.md",
           "Family-level summary (degradation-filtered)",
           "Same as the core table but any family member whose locked "
           "coefficient is flagged as degraded in at least one test seed "
           "is excluded from the family mean/min/max. This isolates "
           "on-manifold steering responses from model-collapse artefacts.")


def main():
    main_table()
    conformist_vs_critical_table()


if __name__ == "__main__":
    main()
