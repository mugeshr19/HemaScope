"""
Agent 8 — Report Generation
Formats Agent 7's synthesis + all agent outputs into a structured PDF
clinical report using reportlab.
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)


@dataclass
class Agent8Result:
    pdf_bytes: bytes
    filename: str

    def to_dict(self) -> dict:
        return {"filename": self.filename, "size_bytes": len(self.pdf_bytes)}


class ReportGenerator:

    def run(
        self,
        prediction_id: str,
        image_name: str,
        agent7_result: dict[str, Any],
        generated_at: str | None = None,
    ) -> Agent9Result:
        generated_at = generated_at or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        buf = io.BytesIO()
        doc = SimpleDocTemplate(
            buf, pagesize=A4,
            leftMargin=20*mm, rightMargin=20*mm,
            topMargin=20*mm, bottomMargin=20*mm,
        )
        story = self._build_story(prediction_id, image_name, agent7_result, generated_at)
        doc.build(story)
        pdf_bytes = buf.getvalue()
        filename = f"hemascope_report_{prediction_id[:8]}.pdf"
        return Agent8Result(pdf_bytes=pdf_bytes, filename=filename)

    # ── Story builder ─────────────────────────────────────────────────────────

    def _build_story(self, prediction_id, image_name, agent7, generated_at):
        styles = getSampleStyleSheet()
        h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontSize=18, spaceAfter=4)
        h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=12, spaceAfter=3, textColor=colors.HexColor("#1a56db"))
        body = ParagraphStyle("body", parent=styles["Normal"], fontSize=9, leading=14)
        small = ParagraphStyle("small", parent=styles["Normal"], fontSize=8, textColor=colors.grey)
        mono = ParagraphStyle("mono", parent=styles["Code"], fontSize=8, leading=12)

        story = []

        # Header
        story += [
            Paragraph("HemaScope — Clinical Blood Smear Report", h1),
            HRFlowable(width="100%", thickness=1, color=colors.HexColor("#1a56db")),
            Spacer(1, 4*mm),
        ]

        # Meta table
        meta = [
            ["Report ID",    prediction_id],
            ["Image",        image_name],
            ["Generated",    generated_at],
            ["LLM Model",    agent7.get("model_used", "N/A")],
        ]
        story.append(self._kv_table(meta))
        story.append(Spacer(1, 6*mm))

        # Agent 1 summary
        a1 = (agent7.get("agent_outputs") or {}).get("agent1", {})
        if a1:
            story.append(Paragraph("Agent 1 — Cell Detection", h2))
            counts = [
                ["Total Cells", "RBC", "WBC", "Platelets"],
                [a1.get("total_cells","—"), a1.get("rbc","—"), a1.get("wbc","—"), a1.get("platelet","—")],
            ]
            story.append(self._data_table(counts))
            story.append(Spacer(1, 4*mm))

        # Agents 2–6 summaries
        sections = [
            ("agent2_malaria",    "Agent 2 — Malaria Screening",   ["risk_level","infected_rbc","total_rbc","parasite_density_pct","recommendation"]),
            ("agent3_morphology", "Agent 3 — RBC Morphology",      ["severity","abnormal_pct","abnormal_count","total_rbc","recommendation"]),
            ("agent4_wbc",        "Agent 4 — WBC Differential",    ["dominant_type","total_wbc","recommendation"]),
            ("agent5_leukemia",   "Agent 5 — Leukemia Screening",  ["risk_level","blast_count","blast_pct","total_wbc","recommendation"]),
            ("agent6_anemia",     "Agent 6 — Anemia Screening",    ["anemia_type","severity","mcv_fl","hb_gdl","recommendation"]),
        ]
        outputs = agent7.get("agent_outputs") or {}
        for key, title, fields in sections:
            data = outputs.get(key)
            if not data:
                continue
            story.append(Paragraph(title, h2))
            rows = [[f, str(data.get(f, "—"))] for f in fields if f in data]
            if rows:
                story.append(self._kv_table(rows))
            story.append(Spacer(1, 4*mm))

        # Agent 8 synthesis
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey))
        story.append(Spacer(1, 3*mm))
        story.append(Paragraph("Agent 7 — Clinical Differential Synthesis", h2))
        synthesis = agent7.get("synthesis", "No synthesis available.")
        for line in synthesis.split("\n"):
            line = line.strip()
            if not line:
                story.append(Spacer(1, 2*mm))
            else:
                story.append(Paragraph(line, body))
        story.append(Spacer(1, 6*mm))

        # Disclaimer
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey))
        story.append(Spacer(1, 2*mm))
        story.append(Paragraph(
            "⚠ DISCLAIMER: This report is generated by an AI system for research and screening purposes only. "
            "It does not constitute a medical diagnosis. All findings must be reviewed and confirmed by a "
            "qualified clinical pathologist or haematologist before any clinical decision is made.",
            small,
        ))

        return story

    # ── Table helpers ─────────────────────────────────────────────────────────

    def _kv_table(self, rows: list) -> Table:
        t = Table(rows, colWidths=[50*mm, 120*mm])
        t.setStyle(TableStyle([
            ("FONTSIZE",    (0, 0), (-1, -1), 9),
            ("FONTNAME",    (0, 0), (0, -1),  "Helvetica-Bold"),
            ("TEXTCOLOR",   (0, 0), (0, -1),  colors.HexColor("#374151")),
            ("BACKGROUND",  (0, 0), (0, -1),  colors.HexColor("#f3f4f6")),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, colors.HexColor("#f9fafb")]),
            ("GRID",        (0, 0), (-1, -1), 0.3, colors.lightgrey),
            ("TOPPADDING",  (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING",(0,0), (-1, -1), 3),
        ]))
        return t

    def _data_table(self, rows: list) -> Table:
        t = Table(rows)
        t.setStyle(TableStyle([
            ("FONTSIZE",    (0, 0), (-1, -1), 9),
            ("FONTNAME",    (0, 0), (-1, 0),  "Helvetica-Bold"),
            ("BACKGROUND",  (0, 0), (-1, 0),  colors.HexColor("#1a56db")),
            ("TEXTCOLOR",   (0, 0), (-1, 0),  colors.white),
            ("ALIGN",       (0, 0), (-1, -1), "CENTER"),
            ("GRID",        (0, 0), (-1, -1), 0.3, colors.lightgrey),
            ("TOPPADDING",  (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING",(0,0), (-1, -1), 4),
        ]))
        return t
