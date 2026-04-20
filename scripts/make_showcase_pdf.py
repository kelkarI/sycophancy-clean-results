"""Typeset a clean PDF of the qualitative tone showcase.

Two representative held-out prompts — Gemma 2 27B on John Locke's
empiricism and Qwen 3 32B on a chemistry professor's false claim —
each decoded under baseline, CAA (targeted), Skeptic (critical), and
Pacifist (conformist) at the tune-locked coefficients. Uses reportlab
for real text layout instead of rasterised matplotlib text.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer,
    Table, TableStyle, PageBreak,
)

HERE = Path(__file__).resolve().parent.parent
QUAL = HERE / "qualitative"
FIGOUT = HERE / "figures"

CRITICAL_BLUE  = colors.HexColor("#08519c")
CONFORMIST_ORG = colors.HexColor("#f16913")
CAA_GRAY       = colors.HexColor("#4d4d4d")
BASELINE_GRAY  = colors.HexColor("#555555")
TEXT_DARK      = colors.HexColor("#1a1a1a")
TEXT_MUTED     = colors.HexColor("#555555")
RULE_LIGHT     = colors.HexColor("#cccccc")
PROMPT_BG      = colors.HexColor("#f4f4f4")

COND_STYLE = {
    "baseline": ("Baseline (no steering)", "no steering",   BASELINE_GRAY),
    "caa":      ("CAA (targeted)",          "targeted",      CAA_GRAY),
    "skeptic":  ("Skeptic",                 "critical role", CRITICAL_BLUE),
    "pacifist": ("Pacifist",                "conformist role", CONFORMIST_ORG),
}


def _styles():
    ss = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "title", parent=ss["Title"], fontName="Times-Bold",
            fontSize=16, leading=20, alignment=TA_LEFT,
            textColor=TEXT_DARK, spaceAfter=2,
        ),
        "subtitle": ParagraphStyle(
            "subtitle", parent=ss["Normal"], fontName="Times-Italic",
            fontSize=10.5, leading=13, textColor=TEXT_MUTED, spaceAfter=14,
        ),
        "model_h": ParagraphStyle(
            "model_h", parent=ss["Normal"], fontName="Times-Bold",
            fontSize=13, leading=16, textColor=TEXT_DARK,
            spaceBefore=8, spaceAfter=4,
        ),
        "prompt_label": ParagraphStyle(
            "prompt_label", parent=ss["Normal"], fontName="Times-Italic",
            fontSize=9, leading=12, textColor=TEXT_MUTED,
        ),
        "prompt_body": ParagraphStyle(
            "prompt_body", parent=ss["Normal"], fontName="Times-Roman",
            fontSize=10, leading=13, textColor=TEXT_DARK,
        ),
        "cond_h": ParagraphStyle(
            "cond_h", parent=ss["Normal"], fontName="Times-Bold",
            fontSize=11.5, leading=14, textColor=colors.white,
        ),
        "cond_h_sub": ParagraphStyle(
            "cond_h_sub", parent=ss["Normal"], fontName="Times-Italic",
            fontSize=8.5, leading=11, textColor=colors.white,
        ),
        "badge": ParagraphStyle(
            "badge", parent=ss["Normal"], fontName="Courier-Bold",
            fontSize=9.5, leading=12, textColor=colors.white,
            alignment=2,  # right
        ),
        "badge_sub": ParagraphStyle(
            "badge_sub", parent=ss["Normal"], fontName="Times-Italic",
            fontSize=8.5, leading=11, textColor=colors.white, alignment=2,
        ),
        "signature": ParagraphStyle(
            "signature", parent=ss["Normal"], fontName="Times-Bold",
            fontSize=11, leading=14, textColor=TEXT_DARK, spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "body", parent=ss["Normal"], fontName="Times-Roman",
            fontSize=10, leading=13.5, textColor=TEXT_DARK, spaceAfter=2,
        ),
        "footer": ParagraphStyle(
            "footer", parent=ss["Normal"], fontName="Times-Italic",
            fontSize=8.5, leading=11, textColor=TEXT_MUTED,
            alignment=1, spaceBefore=12,
        ),
    }


def _clean(s: str) -> str:
    # replace common non-Latin glyphs Times-Roman can't render
    s = s.replace("\u2082", "2").replace("\u2083", "3").replace("\u2081", "1")
    return "".join(c for c in s if ord(c) < 0x2E80).strip()


def _split_response(text: str, head_min: int = 70, head_max: int = 200):
    """Signature opening sentence + continuation paragraph."""
    import re
    honorific = re.compile(r"\b(Mr|Mrs|Ms|Dr|Prof|St|Sr|Jr|H2O|e\.g|i\.e|vs)\.$")
    t = _clean(text).replace("\n\n", "  ").replace("\n", " ")
    t = " ".join(t.split())
    cut = -1
    for m in re.finditer(r"[.!?](?=\s|$)", t):
        end = m.end()
        if end < head_min:
            continue
        if end > head_max:
            break
        if honorific.search(t[:end]):
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
    if len(tail) > 900:
        tail = tail[:900].rsplit(" ", 1)[0] + " …"
    return head, tail


def _response_card(cond, r, metric_kind, styles, content_width):
    label, family, color = COND_STYLE[cond]
    if metric_kind == "syc_logit":
        badge_top = f"syc_logit {r['syc_logit']:+.2f}"
        badge_sub = f"coef {r['coefficient']:+.0f}   A/B pick = {r['chosen']}"
    else:
        badge_top = f"{r['category']}"
        badge_sub = f"coef {r['coefficient']:+.0f}   correct = {r['correct']}"
    if cond == "pacifist" and metric_kind == "category":
        badge_top += "  \u00b7 DEGRADED"

    header_row = Table(
        [[
            [Paragraph(label, styles["cond_h"]),
             Paragraph(family, styles["cond_h_sub"])],
            [Paragraph(badge_top, styles["badge"]),
             Paragraph(badge_sub, styles["badge_sub"])],
        ]],
        colWidths=[content_width * 0.55, content_width * 0.45],
    )
    header_row.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), color),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (0, 0), 10),
        ("RIGHTPADDING", (-1, 0), (-1, 0), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))

    head, tail = _split_response(r["response"])
    sig_style = ParagraphStyle(
        "sig_" + cond, parent=styles["signature"], textColor=color,
    )
    body_flow = [
        Paragraph(_escape(head), sig_style),
        Paragraph(_escape(tail), styles["body"]) if tail else Spacer(1, 0.02 * inch),
    ]

    body_table = Table(
        [[body_flow]],
        colWidths=[content_width],
    )
    body_table.setStyle(TableStyle([
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("BACKGROUND", (0, 0), (-1, -1), colors.white),
        ("LINEBELOW", (0, 0), (-1, -1), 0.5, RULE_LIGHT),
        ("LINELEFT", (0, 0), (-1, -1), 2, color),
    ]))

    return [header_row, body_table, Spacer(1, 0.12 * inch)]


def _escape(s: str) -> str:
    return (s.replace("&", "&amp;")
             .replace("<", "&lt;")
             .replace(">", "&gt;"))


def _prompt_block(prompt_text, styles, content_width):
    body = _clean(prompt_text)
    body = " ".join(body.split())
    t = Table(
        [[[Paragraph("User prompt", styles["prompt_label"]),
           Spacer(1, 2),
           Paragraph(_escape(body), styles["prompt_body"])]]],
        colWidths=[content_width],
    )
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PROMPT_BG),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("BOX", (0, 0), (-1, -1), 0.4, RULE_LIGHT),
    ]))
    return t


def _model_page(model_header, prompt, responses, order, metric_kind,
                styles, content_width):
    flow = [
        Paragraph(model_header, styles["model_h"]),
        _prompt_block(prompt, styles, content_width),
        Spacer(1, 0.18 * inch),
    ]
    for cond in order:
        if cond not in responses:
            continue
        flow.extend(_response_card(cond, responses[cond], metric_kind,
                                   styles, content_width))
    return flow


def main() -> None:
    styles = _styles()

    caa = json.loads((QUAL / "qual_check_caa.json").read_text())
    conf = json.loads((QUAL / "qual_check_conformist.json").read_text())
    gemma_by = defaultdict(dict)
    for r in caa["results"] + conf["results"]:
        gemma_by[r["qi"]][r["condition"]] = r
    gemma_row = gemma_by[2]
    gemma_prompt = gemma_row["baseline"]["prompt_ab"]

    qw = json.loads((QUAL / "qwen3-32b_over_correction.json").read_text())
    qwen_by = defaultdict(dict)
    for r in qw["per_probe_results"]:
        qwen_by[r["topic"]][r["condition"]] = r
    qwen_row = qwen_by["chemistry (false)"]
    qwen_prompt = [p for p in qw["probes"]
                   if p["topic"] == "chemistry (false)"][0]["prompt"]

    order = ["baseline", "caa", "skeptic", "pacifist"]

    page_w, page_h = LETTER
    margin = 0.6 * inch
    frame = Frame(margin, margin, page_w - 2 * margin, page_h - 2 * margin,
                  leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0,
                  id="normal")
    content_width = page_w - 2 * margin

    out = FIGOUT / "fig5_tone_comparison.pdf"
    FIGOUT.mkdir(exist_ok=True)
    doc = BaseDocTemplate(
        str(out), pagesize=LETTER,
        leftMargin=margin, rightMargin=margin,
        topMargin=margin, bottomMargin=margin,
        title="Sycophancy steering — qualitative tone showcase",
        author="sycophancy-clean-results",
    )
    doc.addPageTemplates([PageTemplate(id="main", frames=[frame])])

    flow = [
        Paragraph("How steering changes tone", styles["title"]),
        Paragraph(
            "One representative held-out prompt per model, decoded at each "
            "condition's tune-locked coefficient. Bold = signature opening "
            "sentence. Gemma metric is <font face='Courier'>syc_logit</font> "
            "(= logp(syc) − logp(hon)) on the A/B continuation — lower is "
            "less sycophantic. Qwen metric is the over-correction auto-"
            "category from the sycophancy-qwen pipeline.",
            styles["subtitle"],
        ),
    ]
    flow.extend(_model_page(
        "Gemma 2 27B \u00b7 John Locke's empiricism",
        gemma_prompt, gemma_row, order, "syc_logit",
        styles, content_width,
    ))
    flow.append(PageBreak())
    flow.extend(_model_page(
        "Qwen 3 32B \u00b7 chemistry professor, false claim",
        qwen_prompt, qwen_row, order, "category",
        styles, content_width,
    ))
    flow.append(Paragraph(
        "Full responses: qualitative/gemma_showcase.md, qualitative/qwen_showcase.md. "
        "Degradation flag = collapsed forward pass at the locked coefficient "
        "(rate locks to 0.5, syc-logit gap shrinks to zero, free-form text "
        "becomes a repetition loop).",
        styles["footer"],
    ))

    doc.build(flow)
    print("wrote", out)


if __name__ == "__main__":
    main()
