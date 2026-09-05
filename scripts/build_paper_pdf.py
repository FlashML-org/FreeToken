"""Render the review-copy PDF from the Markdown manuscript.

This deliberately small renderer is intended for draft review, not a venue
submission template. It keeps the manuscript source authoritative and draws
tables plus the bounded-result overview directly from the recorded values.
"""

from __future__ import annotations

import html
import re
import argparse
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "paper-draft" / "amd_strix_halo_freetoken_port_draft.md"
DEFAULT_OUTPUT = ROOT / "output" / "pdf" / "freetoken-amd-strix-halo-white-paper-v0.1.0-rc1.pdf"


def clean(text: str) -> str:
    text = html.escape(text)
    text = re.sub(r"`([^`]+)`", r"<font name='Courier'>\1</font>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\*([^*]+)\*", r"<i>\1</i>", text)
    return text


def page_number(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#B8C2CC"))
    canvas.line(doc.leftMargin, 0.53 * inch, letter[0] - doc.rightMargin, 0.53 * inch)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#53616F"))
    canvas.drawString(doc.leftMargin, 0.35 * inch, "Native FreeToken Serving on AMD Strix Halo")
    canvas.drawRightString(letter[0] - doc.rightMargin, 0.35 * inch, f"Release candidate v0.1.0-rc1 | {doc.page}")
    canvas.restoreState()


def result_overview() -> Table:
    rows = [
        ["Protocol group", "Configuration", "Tokens/s", "Interpretation"],
        ["Qwen NVFP4 canary", "Reference router", "27.88", "Three quality-matched runs"],
        ["Same-file Qwen Q4", "FreeToken baseline", "47.12", "One raw-prompt control"],
        ["Same-file Qwen Q4", "FreeToken plus HIP router", "50.63", "Correct derivation"],
        ["Same-file Qwen Q4", "llama.cpp ROCm 10", "50.29", "Correct derivation"],
        ["Gemma 4 Q4", "FreeToken text control", "57.05", "Fixed arithmetic control"],
    ]
    table = Table(rows, colWidths=[1.40 * inch, 1.75 * inch, 0.65 * inch, 2.55 * inch], repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365D")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.3),
        ("LEADING", (0, 0), (-1, -1), 10),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#AEBBC8")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#EAF0F6")),
        ("BACKGROUND", (0, 2), (-1, 4), colors.HexColor("#F7FAFC")),
        ("BACKGROUND", (0, 5), (-1, 5), colors.HexColor("#EEF5EB")),
    ]))
    return table


def build(output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    title = ParagraphStyle("PaperTitle", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=18, leading=22, alignment=TA_CENTER, textColor=colors.HexColor("#17365D"), spaceAfter=8)
    author = ParagraphStyle("Author", parent=styles["Normal"], fontSize=10, leading=13, alignment=TA_CENTER, textColor=colors.HexColor("#53616F"), spaceAfter=16)
    abstract = ParagraphStyle("Abstract", parent=styles["BodyText"], fontSize=9.2, leading=13, alignment=TA_JUSTIFY, leftIndent=14, rightIndent=14, borderColor=colors.HexColor("#AEBBC8"), borderWidth=0.6, borderPadding=9, spaceAfter=14)
    body = ParagraphStyle("Body", parent=styles["BodyText"], fontName="Helvetica", fontSize=9.2, leading=13, alignment=TA_JUSTIFY, spaceAfter=7)
    h1 = ParagraphStyle("H1", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=13, leading=16, textColor=colors.HexColor("#17365D"), spaceBefore=13, spaceAfter=6, keepWithNext=True)
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=10.5, leading=13, textColor=colors.HexColor("#244F76"), spaceBefore=10, spaceAfter=4, keepWithNext=True)
    small = ParagraphStyle("Small", parent=body, fontSize=8.1, leading=10.5, alignment=TA_LEFT)
    cell = ParagraphStyle("Cell", parent=body, fontSize=7.0, leading=8.4, alignment=TA_LEFT, spaceAfter=0)
    ledger_cell = ParagraphStyle("LedgerCell", parent=body, fontSize=6.2, leading=7.1, alignment=TA_LEFT, spaceAfter=0)
    doc = SimpleDocTemplate(str(output), pagesize=letter, leftMargin=0.72 * inch, rightMargin=0.72 * inch, topMargin=0.62 * inch, bottomMargin=0.72 * inch, title="Native FreeToken Serving on AMD Strix Halo")
    story = []
    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    index = 0
    inserted_overview = False
    while index < len(lines):
        line = lines[index].strip()
        if not line:
            index += 1
            continue
        if line.startswith("# "):
            story.append(Paragraph(clean(line[2:]), title))
        elif line.startswith("**David Bourdeau"):
            story.append(Paragraph(clean(line.strip("*")), author))
        elif line.startswith("*Technical white paper"):
            story.append(Paragraph(clean(line.strip("*")), author))
        elif line == "## Abstract":
            index += 1
            abstract_lines = []
            while index < len(lines) and not lines[index].startswith("## "):
                if lines[index].strip():
                    abstract_lines.append(lines[index].strip())
                index += 1
            story.append(Paragraph("<b>ABSTRACT</b><br/>" + clean(" ".join(abstract_lines)), abstract))
            continue
        elif line.startswith("## "):
            story.append(Paragraph(clean(line[3:]), h1))
        elif line.startswith("### "):
            story.append(Paragraph(clean(line[4:]), h2))
        elif line.startswith("|") and index + 1 < len(lines) and lines[index + 1].startswith("|"):
            table_lines = []
            while index < len(lines) and lines[index].startswith("|"):
                if not re.match(r"^\|\s*[-: ]+\|", lines[index]):
                    table_lines.append([clean(cell.strip()) for cell in lines[index].strip("|").split("|")])
                index += 1
            column_count = len(table_lines[0])
            if column_count == 2:
                widths = [1.48 * inch, 5.07 * inch]
            elif column_count == 3:
                widths = [2.12 * inch, 1.48 * inch, 2.95 * inch]
            elif column_count == 6:
                widths = [1.08 * inch, 1.28 * inch, 0.62 * inch, 0.72 * inch, 1.08 * inch, 1.77 * inch]
            else:
                widths = [6.55 * inch / column_count] * column_count
            rendered_rows = []
            cell_style = ledger_cell if column_count == 3 else cell
            for row_number, row in enumerate(table_lines):
                rendered_rows.append([
                    Paragraph(("<b>" + value + "</b>") if row_number == 0 else value, cell_style)
                    for value in row
                ])
            table = Table(rendered_rows, colWidths=widths, repeatRows=1)
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365D")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#AEBBC8")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ]))
            story.extend([Spacer(1, 4), table])
            if index < len(lines):
                story.append(Spacer(1, 8))
            continue
        elif line.startswith("**Figure 1.") and not inserted_overview:
            story.extend([Spacer(1, 3), result_overview(), Spacer(1, 4), Paragraph(clean(line), small), Spacer(1, 8)])
            inserted_overview = True
        elif line.startswith("**Table "):
            story.append(Paragraph(clean(line), small))
        elif line.startswith("[") and "] " in line:
            story.append(Paragraph(clean(line), small))
        else:
            story.append(Paragraph(clean(line), body))
        index += 1
    doc.build(story, onFirstPage=page_number, onLaterPages=page_number)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render the FreeToken AMD white paper review copy.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="PDF path to create")
    build(parser.parse_args().output)
