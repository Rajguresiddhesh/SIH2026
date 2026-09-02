from __future__ import annotations

import logging
from typing import Optional, Callable

from ..layer2_data_normalization.schema import PackageData, RuleResult, ComplianceDiff
from .rules import ALL_RULES

logger = logging.getLogger(__name__)


class RulebookEngine:
    """
    Orchestrator that runs compliance rules against package data and returns
    a ComplianceDiff with results split into passed/failed/warnings/not_applicable/inconclusive.
    """

    def __init__(self, rules: Optional[list[Callable[[PackageData], RuleResult]]] = None):
        """
        Initialize the RulebookEngine.

        Args:
            rules: List of rule functions to apply. Defaults to ALL_RULES.
        """
        self.rules = rules if rules is not None else ALL_RULES
        logger.info(f"Initialized RulebookEngine with {len(self.rules)} rules.")

    def evaluate(
        self,
        package: PackageData,
        extra_results: Optional[list[RuleResult]] = None,
    ) -> ComplianceDiff:
        """
        Run all rules against the package and return a structured ComplianceDiff.

        Args:
            package: The normalized package data to evaluate.
            extra_results: Already-computed RuleResults to fold into the diff
                (e.g. bar-code / GS1 checks that take extra arguments).

        Returns:
            ComplianceDiff with results categorised into passed/failed/warnings/
            not_applicable/inconclusive lists.
        """
        passed: list[RuleResult] = []
        failed: list[RuleResult] = []
        warnings: list[RuleResult] = []
        not_applicable: list[RuleResult] = []
        inconclusive: list[RuleResult] = []

        results: list[RuleResult] = []
        for rule in self.rules:
            try:
                results.append(rule(package))
            except Exception as e:
                logger.error(f"Error evaluating rule {rule.__name__}: {e}", exc_info=True)
                results.append(RuleResult(
                    rule_id=rule.__name__,
                    rule_name=rule.__name__,
                    status="INCONCLUSIVE",
                    detail=f"Rule evaluation failed with error: {str(e)}",
                    severity="MINOR",
                    weight=0.0,
                ))
        if extra_results:
            results.extend(extra_results)

        for result in results:
            if result.status == "PASS":
                passed.append(result)
            elif result.status == "FAIL":
                failed.append(result)
            elif result.status == "WARNING":
                warnings.append(result)
            elif result.status == "NOT_APPLICABLE":
                not_applicable.append(result)
            else:
                inconclusive.append(result)

        total = len(passed) + len(failed) + len(warnings) + len(not_applicable) + len(inconclusive)

        logger.info(
            "Rulebook evaluation complete — PASS: %d, FAIL: %d, WARN: %d, N/A: %d, INCONCLUSIVE: %d",
            len(passed), len(failed), len(warnings), len(not_applicable), len(inconclusive),
        )

        return ComplianceDiff(
            total_rules=total,
            passed=passed,
            failed=failed,
            warnings=warnings,
            not_applicable=not_applicable,
            inconclusive=inconclusive,
        )

    def evaluate_single(self, package: PackageData, rule_id: str) -> RuleResult:
        """
        Run one specific rule by its rule_id.

        Args:
            package: The package data to evaluate.
            rule_id: The rule_id string to look up.

        Returns:
            RuleResult from that rule.
        """
        for rule in self.rules:
            if rule.__name__ == rule_id:
                return rule(package)
        raise ValueError(f"Rule '{rule_id}' not found in the loaded rulebook.")

    def summary(self, diff: ComplianceDiff) -> str:
        """Generate a human-readable summary of the evaluation."""
        lines = [
            f"Compliance Summary",
            f"{'='*40}",
            f"Total Rules Run   : {diff.total_rules}",
            f"  PASS            : {len(diff.passed)}",
            f"  FAIL            : {len(diff.failed)}",
            f"  WARNING         : {len(diff.warnings)}",
            f"  NOT APPLICABLE  : {len(diff.not_applicable)}",
            f"  INCONCLUSIVE    : {len(diff.inconclusive)}",
        ]

        if diff.failed:
            lines.append("\nViolations:")
            for r in diff.failed:
                lines.append(f"  [{r.severity}] {r.rule_id}: {r.detail}")

        return "\n".join(lines)
