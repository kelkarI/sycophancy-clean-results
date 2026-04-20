"""Typeset a clean PDF of the qualitative tone showcase.

One held-out Gemma 2 27B prompt (John Locke introducing himself as an
empiricist and asking the model to take a side) decoded under every
kept condition at its tune-locked coefficient. No numeric metrics in
the body — just the free-form response and the model's A/B pick on the
paired multiple-choice version of the same prompt.
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
    Table, TableStyle,
)

HERE = Path(__file__).resolve().parent.parent
QUAL = HERE / "qualitative"
FIGOUT = HERE / "figures"

CRITICAL_BLUE  = colors.HexColor("#08519c")
CONFORMIST_ORG = colors.HexColor("#f16913")
CONFORMIST_2   = colors.HexColor("#d94801")
CONFORMIST_3   = colors.HexColor("#fd8d3c")
CAA_GRAY       = colors.HexColor("#4d4d4d")
BASELINE_GRAY  = colors.HexColor("#555555")
TEXT_DARK      = colors.HexColor("#1a1a1a")
TEXT_MUTED     = colors.HexColor("#555555")
RULE_LIGHT     = colors.HexColor("#cccccc")
PROMPT_BG      = colors.HexColor("#f4f4f4")

COND_STYLE = {
    "baseline":     ("Baseline",     "no steering",     BASELINE_GRAY),
    "caa":          ("CAA",           "targeted",        CAA_GRAY),
    "skeptic":      ("Skeptic",       "critical role",   CRITICAL_BLUE),
    "peacekeeper":  ("Peacekeeper",   "conformist role", CONFORMIST_2),
    "pacifist":     ("Pacifist",      "conformist role", CONFORMIST_ORG),
    "collaborator": ("Collaborator",  "conformist role", CONFORMIST_3),
}

ORDER = ["baseline", "caa", "skeptic", "peacekeeper", "pacifist", "collaborator"]


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
        "prompt_label": ParagraphStyle(
            "prompt_label", parent=ss["Normal"], fontName="Times-Italic",
            fontSize=9, leading=12, textColor=TEXT_MUTED,
        ),
        "prompt_body": ParagraphStyle(
            "prompt_body", parent=ss["Normal"], fontName="Times-Roman",
            fontSize=10, leading=13, textColor=TEXT_DARK,
        ),
        "ab_line": ParagraphStyle(
            "ab_line", parent=ss["Normal"], fontName="Times-Roman",
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
        "pick": ParagraphStyle(
            "pick", parent=ss["Normal"], fontName="Courier-Bold",
            fontSize=9.5, leading=12, textColor=colors.white, alignment=2,
        ),
        "pick_sub": ParagraphStyle(
            "pick_sub", parent=ss["Normal"], fontName="Times-Italic",
            fontSize=8.5, leading=11, textColor=colors.white, alignment=2,
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
    s = s.replace("\u2082", "2").replace("\u2083", "3").replace("\u2081", "1")
    return "".join(c for c in s if ord(c) < 0x2E80).strip()


def _escape(s: str) -> str:
    return (s.replace("&", "&amp;")
             .replace("<", "&lt;")
             .replace(">", "&gt;"))


def _normalize(text: str) -> str:
    t = _clean(text).replace("\n\n", "  ").replace("\n", " ")
    return " ".join(t.split())


def _trim(text: str, limit: int = 1100) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + " …"


def _response_card(cond, r, styles, content_width, syc_letter):
    label, family, color = COND_STYLE[cond]
    chosen = r["chosen"]
    agrees = (chosen == syc_letter)
    pick_top = f"Model picked ({chosen})"
    pick_sub = "agrees with user" if agrees else "disagrees with user"

    header_row = Table(
        [[
            [Paragraph(label, styles["cond_h"]),
             Paragraph(family, styles["cond_h_sub"])],
            [Paragraph(pick_top, styles["pick"]),
             Paragraph(pick_sub, styles["pick_sub"])],
        ]],
        colWidths=[content_width * 0.6, content_width * 0.4],
    )
    header_row.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), color),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (0, 0), 10),
        ("RIGHTPADDING", (-1, 0), (-1, 0), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))

    body_text = _trim(_normalize(r["response"]))
    body_table = Table(
        [[Paragraph(_escape(body_text), styles["body"])]],
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

    return [header_row, body_table, Spacer(1, 0.1 * inch)]


def _prompt_block(prompt_text, styles, content_width, syc_letter):
    body = _normalize(prompt_text)
    # Strip the trailing "Answer:" scaffold from prompt_ab for display.
    if body.endswith("Answer:"):
        body = body[: -len("Answer:")].rstrip()
    hint = (
        f"The sycophantic choice here is ({syc_letter}) — it matches the "
        "user's stated view. Each card below shows which option the model "
        "picked under that steering condition."
    )
    t = Table(
        [[[
            Paragraph("User prompt (paired A/B version)", styles["prompt_label"]),
            Spacer(1, 2),
            Paragraph(_escape(body), styles["prompt_body"]),
            Spacer(1, 6),
            Paragraph(_escape(hint), styles["prompt_label"]),
        ]]],
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


def main() -> None:
    styles = _styles()

    caa = json.loads((QUAL / "qual_check_caa.json").read_text())
    conf = json.loads((QUAL / "qual_check_conformist.json").read_text())
    by = defaultdict(dict)
    for r in caa["results"] + conf["results"]:
        by[r["qi"]][r["condition"]] = r
    row = by[2]  # John Locke / empiricism
    prompt = row["baseline"]["prompt_ab"]
    syc_letter = row["baseline"]["sycophantic_answer"]

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
            "Gemma 2 27B decoding the same held-out prompt under every kept "
            "condition at its tune-locked coefficient. The user (\"John "
            "Locke\") introduces himself as an empiricist and asks the model "
            "to take a side — agreeing with the user is the sycophantic "
            "choice. The free-form response is what the model says; the "
            "colored badge shows the A/B pick under the paired multiple-"
            "choice version of the same prompt.",
            styles["subtitle"],
        ),
        _prompt_block(prompt, styles, content_width, syc_letter),
        Spacer(1, 0.18 * inch),
    ]
    for cond in ORDER:
        if cond in row:
            flow.extend(_response_card(cond, row[cond], styles,
                                       content_width, syc_letter))

    flow.append(Paragraph(
        "Critical-role steering (Skeptic) pushes the open response toward "
        "disagreement even when the A/B pick on this single probe still "
        "lands on (A); conformist-role steering (Peacekeeper, Pacifist, "
        "Collaborator) keeps the flattering agreement. Per-base aggregates "
        "across 150 bases \u00d7 3 test seeds are what establish the "
        "family-level effect — see figures/fig1_delta_logit.pdf and "
        "results/main_table.md.",
        styles["footer"],
    ))

    doc.build(flow)
    print("wrote", out)


if __name__ == "__main__":
    main()
