from __future__ import annotations

import json
import logging
from typing import Any, Dict
try:
    from fpdf import FPDF
except ImportError:
    FPDF = None

from ..layer2_data_normalization.schema import ComplianceReport, RuleResult

logger = logging.getLogger(__name__)

class ReportGenerator:
    """Generates PDF and JSON compliance reports."""

    # Characters outside Latin-1 that commonly appear in rule detail strings.
    _UNICODE_MAP = str.maketrans({
        '\u2014': '-',   # em-dash
        '\u2013': '-',   # en-dash
        '\u2012': '-',   # figure dash
        '\u2018': "'",   # left single quotation mark
        '\u2019': "'",   # right single quotation mark
        '\u201c': '"',   # left double quotation mark
        '\u201d': '"',   # right double quotation mark
        '\u2026': '...',  # horizontal ellipsis
        '\u00a0': ' ',   # non-breaking space
        '\u00b0': 'deg', # degree sign (outside some encodings)
        '\u20b9': 'Rs',  # Indian rupee sign
        '\u2192': '->',  # rightwards arrow
        '\u2264': '<=',  # less-than or equal to
        '\u2265': '>=',  # greater-than or equal to
    })

    @staticmethod
    def _safe(text: str) -> str:
        """Transliterate non-Latin-1 characters to ASCII so fpdf core fonts don't choke."""
        translated = text.translate(ReportGenerator._UNICODE_MAP)
        # Final safety net: drop any remaining characters outside Latin-1.
        return translated.encode('latin-1', errors='replace').decode('latin-1')

    def generate_json(self, report: ComplianceReport) -> str:
        """Generate JSON string representation of the report."""
        return report.model_dump_json(indent=2)

    def generate_pdf(self, report: ComplianceReport, output_path: str) -> None:
        """Generate a PDF report."""
        if FPDF is None:
            logger.error("fpdf2 is not installed. Cannot generate PDF. Install with: pip install fpdf2")
            return

        pdf = self._build_pdf(report)
        try:
            pdf.output(output_path)
            logger.info(f"PDF report generated successfully at {output_path}")
        except Exception as e:
            logger.error(f"Error saving PDF report: {e}")

    def _build_pdf(self, report: ComplianceReport) -> Any:
        """Construct the PDF document."""
        pdf = FPDF()
        pdf.add_page()

        # Title Page
        pdf.set_font("helvetica", "B", 16)
        pdf.cell(0, 10, "Legal Metrology Compliance Report", new_x="LMARGIN", new_y="NEXT", align="C")
        pdf.set_font("helvetica", "", 10)
        pdf.cell(0, 10, self._safe(f"Scan ID: {report.scan_id}"), new_x="LMARGIN", new_y="NEXT", align="C")
        pdf.cell(0, 10, self._safe(f"Date: {report.scan_timestamp}"), new_x="LMARGIN", new_y="NEXT", align="C")

        pdf.ln(10)

        # Summary Section
        pdf.set_font("helvetica", "B", 14)
        pdf.cell(0, 10, "Summary", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("helvetica", "", 12)
        score = report.compliance_score

        stars_str = "*" * score.star_rating + "-" * (5 - score.star_rating)
        pdf.cell(0, 10, f"Rating: {stars_str} ({score.star_label})", new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 10, f"Final Score: {score.final_score:.1%}", new_x="LMARGIN", new_y="NEXT")

        na_count = len(report.rulebook_diff.not_applicable) + len(report.rulebook_diff.inconclusive)
        pdf.cell(0, 10, f"Stats: {score.passed_rules} Passed, {score.failed_rules} Failed, {na_count} N/A", new_x="LMARGIN", new_y="NEXT")

        pdf.ln(5)

        # Score Breakdown
        pdf.set_font("helvetica", "B", 12)
        pdf.cell(0, 10, "Score Breakdown", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("helvetica", "", 10)
        pdf.cell(0, 8, f"EBM Assessment Score: {score.ebm_score:.1%}", new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 8, f"Rulebook Assessment Score: {score.rule_score:.1%}", new_x="LMARGIN", new_y="NEXT")

        pdf.ln(5)

        # Rule-by-Rule Results
        pdf.set_font("helvetica", "B", 12)
        pdf.cell(0, 10, "Rule Results", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("helvetica", "", 8)

        # Table Header
        col_widths = [22, 58, 22, 20, 68]
        headers = ["Rule ID", "Name", "Status", "Severity", "Detail"]
        for i, header in enumerate(headers):
            if i < len(headers) - 1:
                pdf.cell(col_widths[i], 7, header, border=1)
            else:
                pdf.cell(col_widths[i], 7, header, border=1, new_x="LMARGIN", new_y="NEXT")

        # Collect all results into one flat list for the table
        all_results: list[RuleResult] = (
            report.rulebook_diff.passed
            + report.rulebook_diff.failed
            + report.rulebook_diff.warnings
            + report.rulebook_diff.not_applicable
            + report.rulebook_diff.inconclusive
        )

        for result in all_results:
            # Colors
            if result.status == "PASS":
                pdf.set_text_color(0, 128, 0)
            elif result.status == "FAIL":
                pdf.set_text_color(220, 0, 0)
            elif result.status == "WARNING":
                pdf.set_text_color(200, 150, 0)
            else:
                pdf.set_text_color(128, 128, 128)

            rule_id_str = self._safe(result.rule_id[:20])
            name_trunc = self._safe((result.rule_name[:28] + "..") if len(result.rule_name) > 30 else result.rule_name)
            detail_trunc = self._safe((result.detail[:38] + "..") if len(result.detail) > 40 else result.detail)

            pdf.cell(col_widths[0], 7, rule_id_str, border=1)
            pdf.cell(col_widths[1], 7, name_trunc, border=1)
            pdf.cell(col_widths[2], 7, self._safe(result.status), border=1)
            pdf.cell(col_widths[3], 7, self._safe(result.severity), border=1)
            pdf.cell(col_widths[4], 7, detail_trunc, border=1, new_x="LMARGIN", new_y="NEXT")

        pdf.set_text_color(0, 0, 0)
        pdf.ln(5)

        # Recommendations
        pdf.set_font("helvetica", "B", 12)
        pdf.cell(0, 10, "Recommendations", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("helvetica", "", 10)
        for i, rec in enumerate(report.recommendations, 1):
            pdf.set_x(pdf.l_margin)  # ensure we're at left margin after table
            pdf.multi_cell(0, 8, self._safe(f"{i}. {rec}"), new_x="LMARGIN", new_y="NEXT")

        pdf.ln(10)
        pdf.set_font("helvetica", "I", 8)
        pdf.cell(0, 10, "Generated by Legal Metrology Compliance System", new_x="LMARGIN", new_y="NEXT", align="C")

        return pdf
