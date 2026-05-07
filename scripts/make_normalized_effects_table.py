"""Emit results/normalized_effects.{csv,md} — one row per (model, condition).

Columns: model | condition | family | alpha | h_baseline_norm |
         perturbation_energy | delta_logit | behavioral_efficiency |
         argmax_cosine_with_caa | argmax_layer

16 rows total (8 conditions × 2 models). Both forms include the
random_0 row so cross-condition ratios are easy to read directly.
"""
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT  = ROOT / "results"
OUT.mkdir(parents=True, exist_ok=True)

MODEL_FILES = [
    ("gemma-2-27b-it", DATA / "gemma-2-27b-it_normalized_effects.json"),
    ("qwen3-32b",      DATA / "qwen3-32b_normalized_effects.json"),
]

CRITICAL = {"skeptic", "devils_advocate", "judge"}
CONFORMIST = {"peacekeeper", "pacifist", "collaborator", "facilitator"}


def _family(c):
    if c == "caa": return "targeted"
    if c == "random_0": return "null"
    if c in CRITICAL: return "critical"
    if c in CONFORMIST: return "conformist"
    return "?"


def _build_rows():
    rows = []
    for model_slug, path in MODEL_FILES:
        if not path.exists():
            print(f"  [skip] {path} missing")
            continue
        d = json.loads(path.read_text())
        h_norm = d["h_baseline_norm_at_target_layer"]
        rand_eff = d["conditions"].get("random_0", {}).get("behavioral_efficiency")
        for cond, c in d["conditions"].items():
            eff = c.get("behavioral_efficiency")
            ratio_vs_random = (eff / rand_eff) if (eff and rand_eff) else None
            rows.append({
                "model": model_slug,
                "condition": cond,
                "family": _family(cond),
                "alpha": c.get("alpha"),
                "h_baseline_norm": round(h_norm, 2),
                "perturbation_energy": (round(c["perturbation_energy"], 4)
                                        if c.get("perturbation_energy") else None),
                "delta_logit": (round(c["delta_logit_mean"], 4)
                                if c.get("delta_logit_mean") is not None else None),
                "delta_logit_ci95_lo": (round(c["delta_logit_ci95_lo"], 4)
                                        if c.get("delta_logit_ci95_lo") is not None else None),
                "delta_logit_ci95_hi": (round(c["delta_logit_ci95_hi"], 4)
                                        if c.get("delta_logit_ci95_hi") is not None else None),
                "behavioral_efficiency": (round(eff, 4) if eff is not None else None),
                "ratio_vs_random_0":     (round(ratio_vs_random, 3) if ratio_vs_random else None),
                "argmax_cosine_with_caa": c.get("argmax_cosine_with_caa"),
                "argmax_layer": c.get("argmax_layer"),
            })
    return rows


def write_csv(rows):
    out = OUT / "normalized_effects.csv"
    if not rows:
        return
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"  wrote {out}  ({len(rows)} rows)")


def write_md(rows):
    out = OUT / "normalized_effects.md"
    if not rows:
        return
    with out.open("w") as f:
        f.write("# Normalized effect sizes (fig11)\n\n")
        f.write("Behavioral efficiency = |∆logit| / ε, where ε = |α| / ‖h̄^baseline_TARGET‖.\n\n")
        f.write("∆logit averaged over 3 test seeds (seed_42, seed_7, seed_123) at the "
                "layerwise experiment's locked α (sourced from each repo's "
                "`results/best_coefs_test.json`). `ratio_vs_random_0` is "
                "behavioral_efficiency / random_0_efficiency on the same model.\n\n")
        f.write(
            "| model | family | condition | α | ‖h̄‖ | ε | Δlogit "
            "| Δlogit CI lo | Δlogit CI hi "
            "| efficiency | ratio vs random | argmax cos | argmax L |\n"
        )
        f.write(
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
        )
        for r in rows:
            row = [
                r["model"], r["family"], r["condition"],
                f"{r['alpha']:+.0f}",
                f"{r['h_baseline_norm']:.0f}",
                f"{r['perturbation_energy']:.4f}",
                (f"{r['delta_logit']:+.4f}" if r["delta_logit"] is not None else ""),
                (f"{r['delta_logit_ci95_lo']:+.4f}" if r["delta_logit_ci95_lo"] is not None else ""),
                (f"{r['delta_logit_ci95_hi']:+.4f}" if r["delta_logit_ci95_hi"] is not None else ""),
                (f"{r['behavioral_efficiency']:.3f}" if r["behavioral_efficiency"] is not None else ""),
                (f"{r['ratio_vs_random_0']:.2f}×" if r["ratio_vs_random_0"] is not None else ""),
                (f"{r['argmax_cosine_with_caa']:+.3f}" if r["argmax_cosine_with_caa"] is not None else ""),
                str(r["argmax_layer"]) if r["argmax_layer"] is not None else "",
            ]
            f.write("| " + " | ".join(row) + " |\n")
    print(f"  wrote {out}  ({len(rows)} rows)")


def main():
    rows = _build_rows()
    write_csv(rows)
    write_md(rows)


if __name__ == "__main__":
    main()
