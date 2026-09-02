from __future__ import annotations

import argparse
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
import uuid
from typing import Any, Optional
import cv2

# Import from all layers
from .layer1_feature_extraction.ocr_engine import OCREngine
from .layer1_feature_extraction.text_parser import TextParser
from .layer1_feature_extraction.object_detector import ObjectDetector
from .layer1_feature_extraction.segmentation import PackageSegmenter
from .layer1_feature_extraction.font_estimator import FontEstimator

from .layer2_data_normalization.normalizer import DataNormalizer
from .layer2_data_normalization.schema import ComplianceReport

from .layer3_ml_model.feature_builder import FeatureBuilder
from .layer3_ml_model.ebm_model import ComplianceEBM
from .layer3_ml_model.training_data import SyntheticDataGenerator

from .layer4_rulebook_engine.engine import RulebookEngine

from .layer5_aggregation.scorer import ComplianceScorer
from .layer5_aggregation.report_generator import ReportGenerator

from .layer1_feature_extraction.gs1 import classify_gtin
from .layer4_rulebook_engine.barcode_rules import evaluate_barcode_rules
from .data_sources.product_lookup import lookup_product
from .layer2_data_normalization.schema import CompliancePrediction

logger = logging.getLogger(__name__)


def run_barcode_pipeline(
    barcode: str,
    *,
    api_key: Optional[str] = None,
    output_path: Optional[str] = None,
    front_image: Optional[str] = None,
    back_image: Optional[str] = None,
) -> ComplianceReport:
    """Deterministic *barcode-first* compliance audit.

    No image or OCR is required.  The bar code number is structurally verified
    as a GS1 GTIN, resolved to product declarations through GS1 India /
    Open Food Facts / other registries, normalised to :class:`PackageData`, and
    run through the full Legal Metrology (Packaged Commodities) Rules, 2011
    rulebook plus the bar-code / GS1 specific checks.

    If a Gemini API key is available the registry data is *enriched* (never
    replaced) with an LLM reading, but the verdict is always produced by the
    deterministic rulebook.
    """
    barcode = str(barcode).strip()
    logger.info("Barcode compliance pipeline — GTIN: %s", barcode)

    gtin_info = classify_gtin(barcode)
    record = lookup_product(barcode)
    logger.info("Registry lookup: found=%s sources=%s", record.found, record.sources)

    llm_used = False
    if api_key:
        try:
            from .llm.llm_compliance_engine import LLMComplianceEngine
            llm_pkg, _, _, _ = LLMComplianceEngine(api_key=api_key).analyze_from_barcode(barcode)
            _enrich_record_from_llm(record, llm_pkg)
            llm_used = True
            logger.info("LLM enrichment applied to registry record.")
        except Exception as e:  # noqa: BLE001
            logger.warning("LLM barcode enrichment failed (%s) — continuing with registry data only.", e)

    normalizer = DataNormalizer()
    package_data = normalizer.from_product_record(record, gtin_info)

    diff = RulebookEngine().evaluate(
        package_data,
        extra_results=evaluate_barcode_rules(package_data, gtin_info),
    )

    passed_w = sum(r.weight for r in diff.passed)
    failed_w = sum(r.weight for r in diff.failed)
    warn_w = sum(r.weight for r in diff.warnings)
    # A bar-code-only audit leaves many mandatory declarations UNVERIFIED
    # (INCONCLUSIVE).  Count those (CRITICAL/MAJOR only) in the denominator so
    # the score honestly reflects how much could actually be confirmed — a
    # product we know nothing about must not read as "100% compliant".
    unverified_w = sum(r.weight for r in diff.inconclusive if r.severity in ("CRITICAL", "MAJOR"))
    applic_w = passed_w + failed_w + warn_w + unverified_w
    rule_prob = (passed_w / applic_w) if applic_w > 0 else 1.0

    contributions = {"barcode_registry_analysis": 1.0}
    if llm_used:
        contributions["llm_barcode_analysis"] = 1.0
    ebm_prediction = CompliancePrediction(
        compliance_probability=rule_prob,
        predicted_compliant=rule_prob >= 0.70,
        feature_contributions=contributions,
        top_risk_factors=[r.rule_name for r in diff.failed][:5],
    )

    scorer = ComplianceScorer()
    # Build the score directly — the bar-code audit's honest figure is
    # "confirmed-compliant weight / (confirmed + unverified-mandatory weight)",
    # not the image-pipeline blend that ignores INCONCLUSIVE results.
    from .layer2_data_normalization.schema import ComplianceScore
    star_rating, star_label = scorer._get_star_rating(rule_prob)
    unverified_mandatory = [r for r in diff.inconclusive if r.severity in ("CRITICAL", "MAJOR")]
    if unverified_mandatory and star_label and "Unverified" not in star_label:
        star_label = f"{star_label} — {len(unverified_mandatory)} mandatory declarations unverified"
    score = ComplianceScore(
        final_score=round(rule_prob, 3),
        ebm_score=round(rule_prob, 3),
        rule_score=round(rule_prob, 3),
        star_rating=star_rating,
        star_label=star_label,
        total_applicable_rules=len(diff.passed) + len(diff.failed) + len(diff.warnings) + len(unverified_mandatory),
        passed_rules=len(diff.passed),
        failed_rules=len(diff.failed),
        critical_failures=sum(1 for r in diff.failed if r.severity == "CRITICAL"),
        major_failures=sum(1 for r in diff.failed if r.severity == "MAJOR"),
        minor_failures=sum(1 for r in diff.failed if r.severity == "MINOR"),
    )
    recommendations = scorer.generate_recommendations(diff, package_data)
    if unverified_mandatory:
        recommendations.insert(0, (
            f"[ACTION] {len(unverified_mandatory)} mandatory declaration(s) could not be verified from "
            f"registry data — physically inspect the pack for: "
            + ", ".join(sorted({r.rule_name for r in unverified_mandatory})) + "."
        ))
    if not record.found:
        recommendations.insert(0, (
            "[INFO] This GTIN was not found in any public product registry. The findings above reflect "
            "only what could be verified from the bar code itself — confirm every mandatory declaration "
            "against the physical package."
        ))

    image_label = f"Barcode: {barcode}"
    imgs = [p for p in (front_image, back_image) if p]
    if imgs:
        image_label = f"{' + '.join(imgs)} (+ barcode {barcode})"

    report = ComplianceReport(
        scan_id=str(uuid.uuid4())[:8],
        scan_timestamp=datetime.now().isoformat(),
        image_path=image_label,
        package_data=package_data,
        ebm_prediction=ebm_prediction,
        rulebook_diff=diff,
        compliance_score=score,
        recommendations=recommendations,
    )

    if output_path:
        gen = ReportGenerator()
        if output_path.endswith(".pdf"):
            gen.generate_pdf(report, output_path)
        elif output_path.endswith(".json"):
            Path(output_path).write_text(gen.generate_json(report))

    return report


def _enrich_record_from_llm(record: Any, llm_pkg: Any) -> None:
    """Fill gaps in a registry :class:`ProductRecord` with an LLM reading.
    Registry values always win; the LLM only supplies fields left blank."""
    mapping = {
        "product_name": "commodity_name",
        "brand": None,
        "manufacturer_name": "manufacturer_name",
        "manufacturer_address": "manufacturer_address",
        "importer_name": "importer_name",
        "importer_address": "importer_address",
        "country_of_origin": "country_of_origin",
        "net_quantity_value": "net_quantity_value",
        "net_quantity_unit": "net_quantity_unit",
        "mrp_value": "mrp_value",
        "manufacture_date": "manufacture_date",
        "expiry_date": "expiry_date",
        "best_before": "best_before",
        "fssai_license": "fssai_license_number",
        "consumer_care_phone": "consumer_care_phone",
        "consumer_care_email": "consumer_care_email",
    }
    for rec_field, pkg_field in mapping.items():
        if pkg_field is None:
            continue
        if getattr(record, rec_field, None) in (None, "", []):
            val = getattr(llm_pkg, pkg_field, None)
            if val not in (None, "", []):
                setattr(record, rec_field, val)
                record.field_sources[rec_field] = "gemini_llm"
    if "gemini_llm" not in record.sources and any(
        v == "gemini_llm" for v in record.field_sources.values()
    ):
        record.sources.append("gemini_llm")


def _process_single_image(
    image_path: str,
    font_est: FontEstimator,
    ocr_engine: OCREngine,
    text_parser: TextParser,
    detector: ObjectDetector,
    segmenter: PackageSegmenter,
    normalizer: DataNormalizer,
    calibration_override=None,
    package_height_mm: Optional[float] = None,
):
    """Run Layer 1 + Layer 2 on one image and return (PackageData, calibration).

    If calibration_override is provided it is used instead of auto-detecting
    from a barcode / package dimension.
    """
    logger.info("Processing image: %s", image_path)

    # Layer 1a: OCR
    ocr_results = ocr_engine.extract_text(image_path)

    # Layer 1b: Text parsing
    parsed = text_parser.parse(ocr_results)

    # Layer 1c: Symbol detection
    symbols = detector.detect(image_path)

    # Layer 1d: Panel segmentation
    panel = segmenter.detect_principal_panel(image_path)

    # Layer 1e: Calibration
    calibration = calibration_override
    if calibration is None:
        img_cv = cv2.imread(image_path)
        if package_height_mm and panel.height_px > 0:
            calibration = font_est.calibrate_from_package_dimension(
                panel.height_px, package_height_mm
            )
        elif img_cv is not None:
            calibration = font_est.calibrate_from_barcode(img_cv)

    # Layer 1f: Font metrics (only when calibration available)
    font_metrics = []
    if calibration:
        panel_area = segmenter.compute_panel_area_mm2(panel, calibration.pixels_per_mm)
        font_metrics = font_est.estimate_font_metrics(ocr_results, calibration, panel_area)
    else:
        logger.warning("No calibration available for %s — font metrics skipped.", image_path)

    # Layer 2: Normalise
    package_data = normalizer.normalize(
        parsed, symbols, panel, font_metrics, ocr_results, calibration
    )

    # Layer 1g: Barcode & QR code scanning
    try:
        from legal_metrology_ml.layer1_feature_extraction.barcode_scanner import BarcodeScanner
        scanner = BarcodeScanner()
        barcodes = scanner.scan(image_path)
        if barcodes:
            bc = barcodes[0]
            package_data.has_barcode = True
            package_data.barcode_value = bc.code
            package_data.barcode_type = bc.type
            package_data.barcode_valid = bc.is_valid
            package_data.barcode_country = bc.country
            logger.info("Detected %s barcode on %s: %s (Country: %s, Valid: %s)",
                        bc.type, Path(image_path).name, bc.code, bc.country, bc.is_valid)
    except Exception as e:
        logger.debug("Barcode scan error on %s: %s", image_path, e)

    return package_data, calibration


def run_pipeline(
    front_image: Optional[str] = None,
    back_image: Optional[str] = None,
    ruler_image: Optional[str] = None,
    barcode_image: Optional[str] = None,
    barcode_number: Optional[str] = None,
    package_height_mm: Optional[float] = None,
    output_path: Optional[str] = None,
    ocr_engine: Optional[OCREngine] = None,
    api_key: Optional[str] = None,
) -> ComplianceReport:
    """Execute the compliance pipeline (LLM-first with local OCR fallback).

    Args:
        front_image:       Path to the front-panel label image (required).
        back_image:        Path to the back/side label image (recommended).
                           Provides manufacturer, net qty, FSSAI, dates, etc.
        ruler_image:       Path to an image showing a mm ruler next to the label.
                           Enables accurate font-size calibration (Rule 7).
        barcode_image:     Optional path to an image of the barcode / QR code.
        barcode_number:    Optional manual barcode number (e.g. EAN-13 digits).
        package_height_mm: Known real-world package height in mm. Alternative
                           calibration method when neither ruler nor barcode
                           is available.
        output_path:       Where to write the output report (.pdf or .json).
        ocr_engine:        Pre-warmed OCREngine instance.
        api_key:           Optional Google Gemini API key.

    Returns:
        ComplianceReport with the merged analysis results.
    """
    logger.info("Starting pipeline — front: %s | back: %s | ruler: %s | barcode: %s",
                front_image, back_image or "—", ruler_image or "—", barcode_image or barcode_number or "—")

    # ── Check for LLM Compliance Engine (Gemini) ────────────────────────────
    from legal_metrology_ml.llm.llm_compliance_engine import LLMComplianceEngine, get_api_key
    resolved_key = get_api_key(api_key)

    # ── Barcode-first pipeline (deterministic; runs with or without Gemini) ──
    active_barcode = str(barcode_number).strip() if barcode_number else None
    if not active_barcode and barcode_image:
        try:
            from legal_metrology_ml.layer1_feature_extraction.barcode_scanner import BarcodeScanner
            bcs = BarcodeScanner().scan(barcode_image)
            if bcs:
                active_barcode = bcs[0].code
        except Exception:
            pass

    if active_barcode and not front_image:
        # Pure barcode audit — no label photo to analyse.
        try:
            return run_barcode_pipeline(
                active_barcode,
                api_key=resolved_key,
                output_path=output_path,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("Barcode compliance pipeline failed (%s).", e)
            raise RuntimeError(f"Barcode compliance inspection failed: {e}")

    if resolved_key:
        llm_engine = LLMComplianceEngine(api_key=resolved_key)

        # Vision LLM pipeline for label photos when no barcode or as fallback
        if front_image:
            logger.info("Using Multimodal Gemini Vision LLM for label photos: front=%s, back=%s", front_image, back_image)
            try:
                pkg_data, score, diff, recs = llm_engine.analyze_from_images(
                    front_image, back_image, package_height_mm=package_height_mm
                )
                from legal_metrology_ml.layer3_ml_model.ebm_model import CompliancePrediction
                ebm_prediction = CompliancePrediction(
                    compliance_probability=score.final_score,
                    predicted_compliant=(score.final_score >= 0.70),
                    feature_contributions={"llm_vision_analysis": 1.0},
                    top_risk_factors=[r.rule_name for r in diff.failed],
                )
                image_label = front_image
                if back_image:
                    image_label = f"{front_image} + {back_image}"

                report = ComplianceReport(
                    scan_id=str(uuid.uuid4())[:8],
                    scan_timestamp=datetime.now().isoformat(),
                    image_path=image_label,
                    package_data=pkg_data,
                    ebm_prediction=ebm_prediction,
                    rulebook_diff=diff,
                    compliance_score=score,
                    recommendations=recs,
                )
                if output_path:
                    gen = ReportGenerator()
                    if output_path.endswith('.pdf'):
                        gen.generate_pdf(report, output_path)
                    elif output_path.endswith('.json'):
                        Path(output_path).write_text(gen.generate_json(report))
                return report
            except Exception as e:
                logger.warning("LLM Vision analysis failed (%s), falling back to local OCR pipeline.", e)

    # ── Shared pipeline components (Local OCR fallback) ─────────────────
    if not front_image:
        raise ValueError("Please provide a packaging label photo, or a bar code / GTIN number to run "
                         "the bar-code compliance audit.")
    # Reuse a pre-built OCREngine if provided (e.g. Flask singleton) so the
    # EasyOCR models are not reloaded on every request.
    if ocr_engine is None:
        ocr_engine = OCREngine()
    text_parser  = TextParser()
    detector     = ObjectDetector()
    segmenter    = PackageSegmenter()
    font_est     = FontEstimator()
    normalizer   = DataNormalizer()

    # ── Ruler calibration (highest priority) ───────────────────────────────
    ruler_calibration = None
    if ruler_image:
        logger.info("Layer 1: Ruler calibration from %s", ruler_image)
        img_ruler = cv2.imread(ruler_image)
        if img_ruler is not None:
            ruler_calibration = font_est.calibrate_from_ruler(img_ruler)
            if ruler_calibration:
                logger.info("Ruler calibration: %.2f px/mm (confidence %.0f%%)",
                            ruler_calibration.pixels_per_mm,
                            ruler_calibration.confidence * 100)
            else:
                logger.warning("Ruler detection failed — will fall back to barcode/dimension.")
        else:
            logger.warning("Could not read ruler image: %s", ruler_image)

    # ── Front image ─────────────────────────────────────────────────────────
    logger.info("Layer 1+2: Front label — %s", front_image)
    front_data, front_cal = _process_single_image(
        front_image, font_est, ocr_engine, text_parser, detector, segmenter,
        normalizer,
        calibration_override=ruler_calibration,
        package_height_mm=package_height_mm,
    )

    # ── Back / side image (optional but recommended) ────────────────────────
    package_data = front_data
    if back_image:
        logger.info("Layer 1+2: Back/side label — %s", back_image)
        # Use the same calibration so font metrics are consistent
        cal_for_back = ruler_calibration or front_cal
        back_data, _ = _process_single_image(
            back_image, font_est, ocr_engine, text_parser, detector, segmenter,
            normalizer,
            calibration_override=cal_for_back,
            package_height_mm=package_height_mm,
        )
        logger.info("Layer 2: Merging front + back PackageData")
        package_data = DataNormalizer.merge(front_data, back_data)
    else:
        logger.info("No back image provided — using front label data only.")

    # ── Explicit Barcode Override / Attachment ──────────────────────────────
    barcode_gtin_info = None
    barcode_record = None
    if not active_barcode and barcode_image:
        try:
            from legal_metrology_ml.layer1_feature_extraction.barcode_scanner import BarcodeScanner
            bcs = BarcodeScanner().scan(barcode_image)
            if bcs:
                active_barcode = bcs[0].code
        except Exception as e:
            logger.warning("Error scanning explicit barcode image: %s", e)
    # A bar code read off the label itself by _process_single_image
    if not active_barcode and package_data.barcode_value:
        active_barcode = package_data.barcode_value

    if active_barcode:
        bc_val = str(active_barcode).strip()
        barcode_gtin_info = classify_gtin(bc_val)
        package_data.has_barcode = True
        package_data.barcode_value = bc_val
        package_data.barcode_type = barcode_gtin_info.fmt or ("EAN-13" if len(bc_val) == 13 else "BARCODE")
        package_data.barcode_gtin_format = barcode_gtin_info.fmt
        package_data.barcode_checksum_valid = barcode_gtin_info.checksum_valid
        package_data.barcode_valid = barcode_gtin_info.is_valid
        package_data.barcode_country = barcode_gtin_info.issuing_country
        package_data.barcode_is_gs1_india = barcode_gtin_info.is_gs1_india
        package_data.barcode_is_restricted = barcode_gtin_info.is_restricted
        try:
            barcode_record = lookup_product(bc_val)
            package_data.product_data_sources = list(barcode_record.sources)
            package_data.product_identified = barcode_record.found
            package_data.barcode_registered_owner = (
                barcode_record.brand or barcode_record.manufacturer_name
            )
            # Fill only fields the label OCR could not extract.
            for pkg_f, rec_f in [
                ("commodity_name", "product_name"),
                ("manufacturer_name", "manufacturer_name"),
                ("manufacturer_address", "manufacturer_address"),
                ("country_of_origin", "country_of_origin"),
                ("net_quantity_value", "net_quantity_value"),
                ("net_quantity_unit", "net_quantity_unit"),
            ]:
                if getattr(package_data, pkg_f) in (None, "", []):
                    v = getattr(barcode_record, rec_f, None)
                    if v not in (None, "", []):
                        setattr(package_data, pkg_f, v)
                        package_data.data_provenance[pkg_f] = "barcode_registry"
        except Exception as e:  # noqa: BLE001
            logger.warning("Registry lookup for attached barcode failed: %s", e)
        logger.info("Barcode attached to image audit: %s (valid GTIN: %s, GS1 India: %s)",
                    bc_val, barcode_gtin_info.is_valid, barcode_gtin_info.is_gs1_india)

    # ── Layer 3: ML Assessment ──────────────────────────────────────────────
    logger.info("Layer 3: ML Assessment")
    builder = FeatureBuilder()
    features_df, feature_types = builder.build(package_data)

    ebm = ComplianceEBM()
    model_path = Path(__file__).parent / 'data' / 'ebm_model' / 'compliance_ebm.pkl'
    if model_path.exists():
        ebm.load(str(model_path))
    else:
        logger.info("Training default EBM model on synthetic data…")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        generator = SyntheticDataGenerator()
        ebm = generator.train_default_model(str(model_path))

    ebm_prediction = ebm.predict(features_df)

    # ── Layer 4: Rulebook ───────────────────────────────────────────────────
    logger.info("Layer 4: Rulebook Evaluation")
    engine = RulebookEngine()
    extra = evaluate_barcode_rules(package_data, barcode_gtin_info) if package_data.has_barcode else None
    rulebook_diff = engine.evaluate(package_data, extra_results=extra)

    # ── Layer 5: Score & Report ─────────────────────────────────────────────
    logger.info("Layer 5: Aggregation")
    scorer = ComplianceScorer()
    score = scorer.compute(ebm_prediction, rulebook_diff)
    recommendations = scorer.generate_recommendations(rulebook_diff, package_data)

    image_label = front_image
    if back_image:
        image_label = f"{front_image} + {back_image}"

    report = ComplianceReport(
        scan_id=str(uuid.uuid4())[:8],
        scan_timestamp=datetime.now().isoformat(),
        image_path=image_label,
        package_data=package_data,
        ebm_prediction=ebm_prediction,
        rulebook_diff=rulebook_diff,
        compliance_score=score,
        recommendations=recommendations,
    )

    # Generate outputs
    if output_path:
        logger.info("Generating output report at %s", output_path)
        gen = ReportGenerator()
        if output_path.endswith('.pdf'):
            gen.generate_pdf(report, output_path)
        elif output_path.endswith('.json'):
            json_str = gen.generate_json(report)
            Path(output_path).write_text(json_str)

    return report


def main():
    parser = argparse.ArgumentParser(description='Legal Metrology Compliance Checker')
    parser.add_argument('--front', required=True,
                        help='Path to the FRONT label image (required)')
    parser.add_argument('--back',
                        help='Path to the BACK / SIDE label image (recommended — '
                             'contains manufacturer, net qty, FSSAI, dates)')
    parser.add_argument('--ruler',
                        help='Path to an image showing a mm ruler (optional — '
                             'enables precise font-size calibration for Rule 7)')
    parser.add_argument('--package-height-mm', type=float,
                        help='Real-world package height in mm (alternative calibration)')
    parser.add_argument('--output', help='Output report path (.pdf or .json)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    report = run_pipeline(
        front_image=args.front,
        back_image=args.back,
        ruler_image=args.ruler,
        package_height_mm=args.package_height_mm,
        output_path=args.output,
    )

    print(f"\n{'='*60}")
    print(f"COMPLIANCE REPORT — {report.scan_id}")
    print(f"{'='*60}")
    stars = '★' * report.compliance_score.star_rating + '☆' * (5 - report.compliance_score.star_rating)
    print(f"Rating: {stars} ({report.compliance_score.star_label})")
    print(f"Score: {report.compliance_score.final_score:.1%}")
    print(f"Rules: {report.compliance_score.passed_rules} passed, {report.compliance_score.failed_rules} failed")

    if report.compliance_score.failed_rules > 0:
        print(f"\nFailures:")
        for result in report.rulebook_diff.failed:
            print(f"  - [{result.severity}] {result.rule_name}: {result.detail}")

    print(f"\nRecommendations:")
    for i, rec in enumerate(report.recommendations, 1):
        print(f"  {i}. {rec}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
