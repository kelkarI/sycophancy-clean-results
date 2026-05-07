"""Emit results/perturbation_propagation.{csv,md} — one row per
(model, persona) pair summarising the per-layer experiment.

Columns:
  model, persona, n_prompts,
  inject_layer, inject_cos_observed, inject_cos_expected, inject_cos_abs_err,
  midpoint_layer, midpoint_cos_mean, midpoint_cos_ci_lo, midpoint_cos_ci_hi,
  final_layer, final_cos_mean, final_cos_ci_lo, final_cos_ci_hi,
  argmax_layer, argmax_cos_mean,
  norm_persona_inject, norm_persona_final, norm_persona_decay,
  norm_caa_inject, norm_caa_final, norm_caa_decay

`*_decay` = norm_final / norm_inject. Values << 1 indicate that the
final-layer cosine reading is being computed on small vectors and
should be interpreted with that caveat (per the README's
"perturbation propagation" section).
"""
import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT  = ROOT / "results"
OUT.mkdir(parents=True, exist_ok=True)

MODEL_FILES = [
    ("gemma-2-27b-it", DATA / "gemma-2-27b-it_perturbation_propagation.json"),
    ("qwen3-32b",      DATA / "qwen3-32b_perturbation_propagation.json"),
]
CRITICAL_PERSONAS = ["skeptic", "devils_advocate", "judge"]


def _midpoint_layer(layers, target):
    """Layer roughly halfway between TARGET and the last captured block."""
    layers = list(layers)
    if not layers:
        return None
    inj_idx = layers.index(target) if target in layers else 0
    final_idx = len(layers) - 1
    mid_idx = (inj_idx + final_idx) // 2
    return layers[mid_idx]


def _build_rows():
    rows = []
    for model_slug, path in MODEL_FILES:
        if not path.exists():
            print(f"  [skip] {path} missing")
            continue
        d = json.loads(path.read_text())
        layers = list(d["layers"])
        target = int(d["target_layer"])
        n_prompts = int(d["n_prompts"])
        inj_idx = layers.index(target)
        final_idx = len(layers) - 1
        final_layer = layers[final_idx]
        midpoint = _midpoint_layer(layers, target)
        mid_idx = layers.index(midpoint)

        # Norm decay tables (per condition).
        norm_block = d.get("norm", {})

        for persona in d["personas_compared_to_caa"]:
            cs = d["cosine"][persona]
            mean = np.array(cs["mean"])
            ci_lo = np.array(cs["ci_lo"])
            ci_hi = np.array(cs["ci_hi"])
            argmax_idx = int(np.argmax(np.abs(mean)))

            sanity = d.get("injection_layer_sanity_check", {}).get(persona, {})
            inj_obs = float(sanity.get("observed", float(mean[inj_idx])))
            inj_exp = sanity.get("expected", None)
            inj_err = sanity.get("abs_err", None)

            def _norm_at(cond, layer_idx):
                if cond not in norm_block:
                    return None
                m = norm_block[cond]["mean"]
                if layer_idx >= len(m):
                    return None
                return float(m[layer_idx])

            np_inj = _norm_at(persona, inj_idx)
            np_fin = _norm_at(persona, final_idx)
            nc_inj = _norm_at("caa", inj_idx)
            nc_fin = _norm_at("caa", final_idx)
            np_decay = (np_fin / np_inj) if (np_inj and np_fin) else None
            nc_decay = (nc_fin / nc_inj) if (nc_inj and nc_fin) else None

            rows.append({
                "model": model_slug,
                "persona": persona,
                "n_prompts": n_prompts,
                "inject_layer": target,
                "inject_cos_observed": round(inj_obs, 4),
                "inject_cos_expected": (round(inj_exp, 4)
                                        if inj_exp is not None else None),
                "inject_cos_abs_err":  (round(inj_err, 4)
                                        if inj_err is not None else None),
                "midpoint_layer": midpoint,
                "midpoint_cos_mean":   round(float(mean[mid_idx]), 4),
                "midpoint_cos_ci_lo":  round(float(ci_lo[mid_idx]), 4),
                "midpoint_cos_ci_hi":  round(float(ci_hi[mid_idx]), 4),
                "final_layer": final_layer,
                "final_cos_mean":  round(float(mean[final_idx]), 4),
                "final_cos_ci_lo": round(float(ci_lo[final_idx]), 4),
                "final_cos_ci_hi": round(float(ci_hi[final_idx]), 4),
                "argmax_layer":    layers[argmax_idx],
                "argmax_cos_mean": round(float(mean[argmax_idx]), 4),
                "norm_persona_inject": (round(np_inj, 2) if np_inj else None),
                "norm_persona_final":  (round(np_fin, 2) if np_fin else None),
                "norm_persona_decay":  (round(np_decay, 3) if np_decay else None),
                "norm_caa_inject":     (round(nc_inj, 2) if nc_inj else None),
                "norm_caa_final":      (round(nc_fin, 2) if nc_fin else None),
                "norm_caa_decay":      (round(nc_decay, 3) if nc_decay else None),
            })
    return rows


def write_csv(rows):
    if not rows:
        print("  [WARN] no rows to write")
        return
    out = OUT / "perturbation_propagation.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"  wrote {out}  ({len(rows)} rows)")


def write_md(rows):
    if not rows:
        return
    out = OUT / "perturbation_propagation.md"
    with out.open("w") as f:
        f.write("# Perturbation propagation (per-layer ΔH cosine + norm)\n\n")
        f.write("One row per (model, persona). Cosines are the per-layer "
                "mean of ⟨ΔH^CAA, ΔH^persona⟩ averaged within prompt over "
                "tokens then across prompts. `_decay` = norm_final / "
                "norm_inject; values << 1 mean the final-layer cosine is "
                "being read off small vectors.\n\n")
        f.write("| model | persona | n | inj_L | inj_obs | inj_exp | inj_err "
                "| mid_L | mid | mid_lo | mid_hi | fin_L | fin | fin_lo | fin_hi "
                "| argmax_L | argmax | npp_inj | npp_fin | npp_decay "
                "| ncaa_inj | ncaa_fin | ncaa_decay |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            row = [
                r["model"], r["persona"], str(r["n_prompts"]),
                str(r["inject_layer"]),
                f"{r['inject_cos_observed']:+.4f}",
                ("" if r["inject_cos_expected"] is None
                 else f"{r['inject_cos_expected']:+.4f}"),
                ("" if r["inject_cos_abs_err"] is None
                 else f"{r['inject_cos_abs_err']:.4f}"),
                str(r["midpoint_layer"]),
                f"{r['midpoint_cos_mean']:+.4f}",
                f"{r['midpoint_cos_ci_lo']:+.4f}",
                f"{r['midpoint_cos_ci_hi']:+.4f}",
                str(r["final_layer"]),
                f"{r['final_cos_mean']:+.4f}",
                f"{r['final_cos_ci_lo']:+.4f}",
                f"{r['final_cos_ci_hi']:+.4f}",
                str(r["argmax_layer"]),
                f"{r['argmax_cos_mean']:+.4f}",
                ("" if r["norm_persona_inject"] is None
                 else f"{r['norm_persona_inject']:.2f}"),
                ("" if r["norm_persona_final"] is None
                 else f"{r['norm_persona_final']:.2f}"),
                ("" if r["norm_persona_decay"] is None
                 else f"{r['norm_persona_decay']:.3f}"),
                ("" if r["norm_caa_inject"] is None
                 else f"{r['norm_caa_inject']:.2f}"),
                ("" if r["norm_caa_final"] is None
                 else f"{r['norm_caa_final']:.2f}"),
                ("" if r["norm_caa_decay"] is None
                 else f"{r['norm_caa_decay']:.3f}"),
            ]
            f.write("| " + " | ".join(row) + " |\n")
    print(f"  wrote {out}  ({len(rows)} rows)")


def main():
    rows = _build_rows()
    write_csv(rows)
    write_md(rows)


if __name__ == "__main__":
    main()
