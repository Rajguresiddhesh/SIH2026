from __future__ import annotations

import logging
from typing import List, Tuple

from ..layer2_data_normalization.schema import (
    CompliancePrediction, ComplianceDiff, ComplianceScore, RuleResult, PackageData
)

logger = logging.getLogger(__name__)

class ComplianceScorer:
    """
    Combines EBM prediction and rulebook diff into a final compliance score.

    Final Score = α × EBM_compliance_prob + (1-α) × Rule_qualification_rate

    Where Rule_qualification_rate = Σ(passed_rule_weight) / Σ(applicable_rule_weight)
    """

    def __init__(self, ebm_alpha: float = 0.3):
        """
        Initialize the scorer.

        Args:
            ebm_alpha: Weight given to the EBM model prediction (0 to 1).
                       Remaining weight (1 - ebm_alpha) goes to the rulebook score.
        """
        self.ebm_alpha = ebm_alpha

    def compute(self, ebm_prediction: CompliancePrediction, rulebook_diff: ComplianceDiff) -> ComplianceScore:
        """
        Compute the final compliance score based on EBM and Rulebook results.
        """
        # Calculate rulebook score from the list-based ComplianceDiff
        passed_weight = sum(r.weight for r in rulebook_diff.passed)
        failed_weight = sum(r.weight for r in rulebook_diff.failed)
        warning_weight = sum(r.weight for r in rulebook_diff.warnings)

        applicable_weight = passed_weight + failed_weight + warning_weight
        rule_score = (passed_weight / applicable_weight) if applicable_weight > 0 else 1.0
        ebm_score = ebm_prediction.compliance_probability

        # Combine
        final_score = self.ebm_alpha * ebm_score + (1 - self.ebm_alpha) * rule_score

        star_rating, star_label = self._get_star_rating(final_score)
        critical, major, minor = self._count_failures_by_severity(rulebook_diff)

        passed_rules = len(rulebook_diff.passed)
        failed_rules = len(rulebook_diff.failed)
        total_applicable = passed_rules + failed_rules + len(rulebook_diff.warnings)

        return ComplianceScore(
            final_score=final_score,
            ebm_score=ebm_score,
            rule_score=rule_score,
            star_rating=star_rating,
            star_label=star_label,
            total_applicable_rules=total_applicable,
            passed_rules=passed_rules,
            failed_rules=failed_rules,
            critical_failures=critical,
            major_failures=major,
            minor_failures=minor,
        )

    def _count_failures_by_severity(self, diff: ComplianceDiff) -> Tuple[int, int, int]:
        """Returns count of (critical, major, minor) failures."""
        critical = sum(1 for r in diff.failed if r.severity == "CRITICAL")
        major = sum(1 for r in diff.failed if r.severity == "MAJOR")
        minor = sum(1 for r in diff.failed if r.severity == "MINOR")
        return critical, major, minor

    def _get_star_rating(self, score: float) -> Tuple[int, str]:
        """Maps score to star rating and label."""
        if score >= 0.95:
            return 5, "Fully Compliant"
        elif score >= 0.80:
            return 4, "Mostly Compliant"
        elif score >= 0.60:
            return 3, "Partially Compliant"
        elif score >= 0.40:
            return 2, "Largely Non-Compliant"
        else:
            return 1, "Critically Non-Compliant"

    def generate_recommendations(self, diff: ComplianceDiff, package_data: PackageData) -> List[str]:
        """Generate actionable recommendations based on failures."""
        recommendations = []

        for result in diff.failed:
            ref = f" (Ref: {result.legal_reference})" if result.legal_reference else ""
            evidence = f" Found: {result.evidence}" if result.evidence else ""
            rec = f"[{result.severity}] {result.rule_name}{ref}: {result.detail}.{evidence}"
            recommendations.append(rec)

        for result in diff.warnings:
            rec = f"[WARNING] {result.rule_name}: {result.detail}."
            recommendations.append(rec)

        if not recommendations:
            recommendations.append("No compliance issues found. Maintain current packaging standards.")

        return recommendations
