"""Split-conformal prediction for per-rule compliance decisions.

For each rule we have a soft score s in [0,1] = P(FAIL). On a held-out
*calibration* set with known gold verdicts we pick two thresholds
(tau_low, tau_high) so that, at target error rate alpha:

    predict PASS  if s <= tau_low          (guaranteed FP rate <= alpha)
    predict FAIL  if s >= tau_high         (guaranteed FN rate <= alpha)
    else          ABSTAIN  (-> INCONCLUSIVE)

Thresholds are the alpha-quantiles of the calibration nonconformity scores
(Vovk; Angelopoulos & Bates, "A Gentle Introduction to Conformal Prediction").
Guarantees are marginal over the exchangeable calibration+test distribution.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path


def _quantile(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    # conformal correction: ceil((n+1)(1-q))/n  index
    n = len(xs)
    k = math.ceil((n + 1) * q)
    k = min(max(k, 1), n)
    return xs[k - 1]


@dataclass
class RuleConformal:
    rule_id: str
    alpha: float = 0.1
    tau_low: float = 0.0
    tau_high: float = 1.0
    n_cal: int = 0

    def predict(self, p_fail: float) -> str:
        if p_fail <= self.tau_low:
            return "PASS"
        if p_fail >= self.tau_high:
            return "FAIL"
        return "ABSTAIN"


@dataclass
class ConformalCompliance:
    alpha: float = 0.1
    rules: dict[str, RuleConformal] = field(default_factory=dict)

    def calibrate(self, cal: list[tuple[str, str, float]]) -> "ConformalCompliance":
        """cal = list of (rule_id, gold_status in {PASS,FAIL}, p_fail)."""
        by_rule: dict[str, dict[str, list[float]]] = {}
        for rid, gold, p in cal:
            if gold not in ("PASS", "FAIL"):
                continue
            by_rule.setdefault(rid, {"PASS": [], "FAIL": []})[gold].append(float(p))

        for rid, groups in by_rule.items():
            pass_scores = groups["PASS"]      # want these <= tau_low
            fail_scores = groups["FAIL"]      # want these >= tau_high
            # tau_low = (1-alpha) quantile of PASS scores  -> at most alpha of PASS above it
            tau_low = _quantile(pass_scores, 1.0 - self.alpha) if pass_scores else 0.0
            # tau_high = alpha quantile of FAIL scores  -> at most alpha of FAIL below it
            tau_high = _quantile(fail_scores, self.alpha) if fail_scores else 1.0
            if tau_high < tau_low:  # overlap — widen the abstention band symmetrically
                mid = (tau_low + tau_high) / 2
                tau_low, tau_high = mid, mid
            self.rules[rid] = RuleConformal(
                rule_id=rid, alpha=self.alpha,
                tau_low=round(tau_low, 4), tau_high=round(tau_high, 4),
                n_cal=len(pass_scores) + len(fail_scores),
            )
        return self

    def predict(self, rid: str, p_fail: float) -> str:
        rc = self.rules.get(rid)
        if rc is None:
            return "ABSTAIN"
        return rc.predict(p_fail)

    def predict_report(self, scores: dict[str, float]) -> dict[str, str]:
        return {rid: self.predict(rid, p) for rid, p in scores.items()}

    # ── evaluate the guarantee on a test set ────────────────────────────────
    def audit(self, test: list[tuple[str, str, float]]) -> dict:
        out: dict[str, dict] = {}
        for rid in self.rules:
            rows = [(g, p) for r, g, p in test if r == rid and g in ("PASS", "FAIL")]
            if not rows:
                continue
            preds = [(g, self.predict(rid, p)) for g, p in rows]
            covered = [x for x in preds if x[1] in ("PASS", "FAIL")]
            fp = sum(1 for g, pr in covered if g == "PASS" and pr == "FAIL")
            fn = sum(1 for g, pr in covered if g == "FAIL" and pr == "PASS")
            n_pass = sum(1 for g, _ in rows if g == "PASS")
            n_fail = sum(1 for g, _ in rows if g == "FAIL")
            out[rid] = {
                "target_alpha": self.alpha,
                "empirical_fp_rate": round(fp / n_pass, 4) if n_pass else None,
                "empirical_fn_rate": round(fn / n_fail, 4) if n_fail else None,
                "coverage": round(len(covered) / len(rows), 4),
                "tau_low": self.rules[rid].tau_low,
                "tau_high": self.rules[rid].tau_high,
            }
        return out

    def save(self, path: str) -> None:
        Path(path).write_text(
            json.dumps(
                {"alpha": self.alpha,
                 "rules": {k: v.__dict__ for k, v in self.rules.items()}},
                indent=2,
            )
        )

    @classmethod
    def load(cls, path: str) -> "ConformalCompliance":
        d = json.loads(Path(path).read_text())
        cc = cls(alpha=d["alpha"])
        cc.rules = {k: RuleConformal(**v) for k, v in d["rules"].items()}
        return cc


if __name__ == "__main__":
    import random

    rng = random.Random(0)
    # synthetic: FAIL rules have higher p_fail, with noise
    cal, test = [], []
    for split, bucket in (("cal", cal), ("test", test)):
        for _ in range(400):
            gold = rng.choice(["PASS", "FAIL"])
            p = (0.75 if gold == "FAIL" else 0.25) + rng.uniform(-0.35, 0.35)
            bucket.append(("R06_MRP", gold, min(1.0, max(0.0, p))))
    cc = ConformalCompliance(alpha=0.1).calibrate(cal)
    print("thresholds:", cc.rules["R06_MRP"].__dict__)
    print("audit:", json.dumps(cc.audit(test), indent=2))
