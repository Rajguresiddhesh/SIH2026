"""
train_on_real_images.py
=======================
Trains the Legal Metrology EBM compliance model on **real** product label images
found in the ``dataset/`` folder.

Workflow
--------
1. For each image in ``dataset/``:
   a. Run the full Layer 1 pipeline (OCR, text parse, object detection, segmentation,
      font estimation) to produce a ``PackageData`` object.
   b. Run the Layer 4 ``RulebookEngine`` to auto-generate a compliance label:
      - 1  (compliant)   — zero CRITICAL-severity failures
      - 0  (non-compliant) — at least one CRITICAL failure
   c. Convert ``PackageData`` → flat feature vector via ``FeatureBuilder``.

2. Combine the real feature rows with 5 000 synthetic samples from
   ``SyntheticDataGenerator``.

3. Retrain ``ComplianceEBM`` on the combined dataset.

4. Save the model to:
       legal_metrology_ml/data/ebm_model/compliance_ebm.pkl

Usage
-----
    python train_on_real_images.py [--dataset-dir dataset] [--n-synthetic 5000] [--verbose]

Run from the project root (the ``Sheet`` folder that contains ``app.py``).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Make sure the project root is on sys.path so relative imports inside the
# ``legal_metrology_ml`` package resolve correctly when this script is run
# directly (i.e. not via ``python -m``).
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Layer 1 — Feature Extraction
from legal_metrology_ml.layer1_feature_extraction.ocr_engine import OCREngine
from legal_metrology_ml.layer1_feature_extraction.text_parser import TextParser
from legal_metrology_ml.layer1_feature_extraction.object_detector import ObjectDetector
from legal_metrology_ml.layer1_feature_extraction.segmentation import PackageSegmenter
from legal_metrology_ml.layer1_feature_extraction.font_estimator import FontEstimator

# Layer 2 — Normalisation
from legal_metrology_ml.layer2_data_normalization.normalizer import DataNormalizer

# Layer 3 — ML
from legal_metrology_ml.layer3_ml_model.feature_builder import FeatureBuilder
from legal_metrology_ml.layer3_ml_model.ebm_model import ComplianceEBM
from legal_metrology_ml.layer3_ml_model.training_data import (
    SyntheticDataGenerator,
    label_from_rulebook_diff,
)

# Layer 4 — Rulebook
from legal_metrology_ml.layer4_rulebook_engine.engine import RulebookEngine

# ---------------------------------------------------------------------------
# Supported image extensions
# ---------------------------------------------------------------------------
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )


def process_image(
    image_path: Path,
    ocr_engine: OCREngine,
    text_parser: TextParser,
    detector: ObjectDetector,
    segmenter: PackageSegmenter,
    font_estimator: FontEstimator,
    normalizer: DataNormalizer,
    feature_builder: FeatureBuilder,
    rulebook: RulebookEngine,
    logger: logging.Logger,
) -> tuple[pd.DataFrame | None, int | None]:
    """Run the full pipeline on one image.

    Returns
    -------
    (feature_df, label) on success, or (None, None) on failure.
    """
    img_str = str(image_path)
    logger.info("Processing: %s", image_path.name)

    try:
        # ---- Layer 1a: OCR ----
        ocr_results = ocr_engine.extract_text(img_str)
        logger.debug("  OCR: %d text blocks", len(ocr_results))

        # ---- Layer 1b: Text parsing ----
        parsed = text_parser.parse(ocr_results)

        # ---- Layer 1c: Symbol / object detection ----
        symbols = detector.detect(img_str)
        logger.debug("  Symbols detected: %d", len(symbols))

        # ---- Layer 1d: Panel segmentation ----
        panel = segmenter.detect_principal_panel(img_str)

        # ---- Layer 1e: Font estimation ----
        img_cv = cv2.imread(img_str)
        calibration = None
        if img_cv is not None:
            calibration = font_estimator.calibrate_from_barcode(img_cv)

        font_metrics: list = []
        if calibration:
            panel_area = segmenter.compute_panel_area_mm2(panel, calibration.pixels_per_mm)
            font_metrics = font_estimator.estimate_font_metrics(
                ocr_results, calibration, panel_area
            )
        else:
            logger.warning("  Could not calibrate from barcode — font metrics will be NaN")

        # ---- Layer 2: Normalise ----
        package_data = normalizer.normalize(
            parsed, symbols, panel, font_metrics, ocr_results, calibration
        )

        # ---- Layer 4: Auto-label via rulebook ----
        diff = rulebook.evaluate(package_data)
        label = label_from_rulebook_diff(diff)
        critical_fails = [r for r in diff.failed if r.severity == "CRITICAL"]
        logger.info(
            "  Label → %s  (CRITICAL failures: %d, MAJOR: %d, MINOR: %d)",
            "COMPLIANT" if label == 1 else "NON-COMPLIANT",
            len(critical_fails),
            len([r for r in diff.failed if r.severity == "MAJOR"]),
            len([r for r in diff.failed if r.severity == "MINOR"]),
        )
        if critical_fails:
            for r in critical_fails:
                logger.info("    ✗ [CRITICAL] %s: %s", r.rule_id, r.detail)

        # ---- Layer 3a: Build feature vector ----
        feat_df, _ = feature_builder.build(package_data)
        # Align columns to the exact order expected by the model
        feat_df = feat_df[feature_builder.FEATURE_NAMES]
        return feat_df, label

    except Exception as exc:
        logger.error("  FAILED to process %s: %s", image_path.name, exc, exc_info=True)
        return None, None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the Legal Metrology EBM on real dataset images"
    )
    parser.add_argument(
        "--dataset-dir",
        default="dataset",
        help="Directory containing real product label images (default: dataset)",
    )
    parser.add_argument(
        "--n-synthetic",
        type=int,
        default=5000,
        help="Number of synthetic training samples to mix in (default: 5000)",
    )
    parser.add_argument(
        "--noise-level",
        type=float,
        default=0.1,
        help="Noise level for synthetic data generation (default: 0.1)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable DEBUG logging")
    args = parser.parse_args()

    setup_logging(args.verbose)
    log = logging.getLogger("train_on_real_images")

    # ------------------------------------------------------------------
    # Locate dataset images
    # ------------------------------------------------------------------
    dataset_dir = PROJECT_ROOT / args.dataset_dir
    if not dataset_dir.exists():
        log.error("Dataset directory not found: %s", dataset_dir)
        sys.exit(1)

    image_paths = sorted(
        p for p in dataset_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        log.error("No images found in %s", dataset_dir)
        sys.exit(1)

    log.info("Found %d image(s) in '%s'", len(image_paths), dataset_dir)

    # ------------------------------------------------------------------
    # Initialise pipeline components (once — they are reused per image)
    # ------------------------------------------------------------------
    log.info("Initialising pipeline components…")
    ocr_engine     = OCREngine()
    text_parser    = TextParser()
    detector       = ObjectDetector()
    segmenter      = PackageSegmenter()
    font_estimator = FontEstimator()
    normalizer     = DataNormalizer()
    feature_builder = FeatureBuilder()
    rulebook       = RulebookEngine()

    # ------------------------------------------------------------------
    # Process each image
    # ------------------------------------------------------------------
    real_rows: list[pd.DataFrame] = []
    real_labels: list[int] = []

    for img_path in image_paths:
        feat_df, label = process_image(
            img_path,
            ocr_engine,
            text_parser,
            detector,
            segmenter,
            font_estimator,
            normalizer,
            feature_builder,
            rulebook,
            log,
        )
        if feat_df is not None:
            real_rows.append(feat_df)
            real_labels.append(label)

    n_real = len(real_rows)
    log.info("\n%s", "=" * 60)
    log.info("Real images processed successfully: %d / %d", n_real, len(image_paths))

    if n_real == 0:
        log.error("No images were processed successfully. Cannot train. Exiting.")
        sys.exit(1)

    real_X = pd.concat(real_rows, ignore_index=True)
    real_y = pd.Series(real_labels, dtype=int)

    real_label_counts = real_y.value_counts()
    log.info(
        "Real label distribution — COMPLIANT: %d  |  NON-COMPLIANT: %d",
        real_label_counts.get(1, 0),
        real_label_counts.get(0, 0),
    )

    # ------------------------------------------------------------------
    # Generate synthetic data
    # ------------------------------------------------------------------
    log.info("Generating %d synthetic samples…", args.n_synthetic)
    generator = SyntheticDataGenerator()
    synth_X, synth_y = generator.generate(
        n_samples=args.n_synthetic, noise_level=args.noise_level
    )
    synth_X = synth_X[feature_builder.FEATURE_NAMES]

    # ------------------------------------------------------------------
    # Combine real + synthetic
    # ------------------------------------------------------------------
    combined_X = pd.concat([real_X, synth_X], ignore_index=True)
    combined_y = pd.concat([real_y, synth_y], ignore_index=True)

    log.info(
        "Combined dataset — total: %d  (real: %d, synthetic: %d)",
        len(combined_X),
        n_real,
        args.n_synthetic,
    )
    log.info(
        "Combined label distribution:\n%s",
        combined_y.value_counts(normalize=True).to_string(),
    )

    # ------------------------------------------------------------------
    # Get feature types from FeatureBuilder
    # ------------------------------------------------------------------
    _, feature_types = feature_builder.build(
        __import__(
            "legal_metrology_ml.layer2_data_normalization.schema",
            fromlist=["PackageData"],
        ).PackageData()
    )

    # ------------------------------------------------------------------
    # Train the EBM
    # ------------------------------------------------------------------
    log.info("Training EBM model…")
    ebm = ComplianceEBM()
    metrics = ebm.train(combined_X, combined_y, feature_types=feature_types)
    log.info("Training complete. Accuracy on combined data: %.4f", metrics["accuracy"])

    # ------------------------------------------------------------------
    # Save the model
    # ------------------------------------------------------------------
    model_path = PROJECT_ROOT / "legal_metrology_ml" / "data" / "ebm_model" / "compliance_ebm.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    ebm.save(str(model_path))

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  EBM TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Real images used      : {n_real}")
    print(f"  Synthetic samples     : {args.n_synthetic}")
    print(f"  Total training rows   : {len(combined_X)}")
    print(f"  Training accuracy     : {metrics['accuracy']:.2%}")
    print(f"  Model saved to        : {model_path.relative_to(PROJECT_ROOT)}")
    print("=" * 60)

    print("\nReal-image label breakdown:")
    for img_path, label in zip(
        [p for p in image_paths if p.suffix.lower() in IMAGE_EXTENSIONS],
        real_labels,
    ):
        status = "✔ COMPLIANT" if label == 1 else "✘ NON-COMPLIANT"
        print(f"  {img_path.name:<55} → {status}")

    print("\nGlobal feature importances (top 10):")
    try:
        importances = ebm.explain_global()
        top10 = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10]
        for feat, score in top10:
            bar = "█" * int(score * 30 / max(s for _, s in top10))
            print(f"  {feat:<35} {bar} {score:.4f}")
    except Exception as exc:
        log.warning("Could not extract global importances: %s", exc)

    print()


if __name__ == "__main__":
    main()
