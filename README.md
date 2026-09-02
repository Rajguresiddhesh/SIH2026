# Legal Metrology ML Pipeline

A comprehensive machine learning and rule-based system for evaluating compliance of packaged products according to Legal Metrology standards.

## Architecture

This pipeline is organized into 5 layers:

1. **Layer 1: Feature Extraction**: Processes raw images to extract text (OCR), detect standard symbols, segment panels, and estimate real-world font metrics based on dimension calibration.
2. **Layer 2: Data Normalization**: Takes extracted raw inputs and maps them onto a standardized, well-typed schema (`NormalizedPackageData`).
3. **Layer 3: ML Model**: Uses an Explainable Boosting Machine (EBM) model to score the probability of compliance from a holistic perspective.
4. **Layer 4: Rulebook Engine**: Performs deterministic, rule-based checks applying legal metrology regulations (e.g., minimum font size tables, mandatory fields).
5. **Layer 5: Aggregation & Reporting**: Combines ML predictions with rulebook diffs to generate a final `ComplianceScore`, along with actionable recommendations. Outputs JSON or PDF reports.

## Installation

Ensure you have Python 3.10+ installed.

```bash
pip install -r requirements.txt
```

## Usage

```bash
python -m legal_metrology_ml.main --image data/sample_images/packaging_1.jpg --output report.pdf
```

### Options

- `--image`: Path to the product image (Required)
- `--package-height-mm`: Real-world package height for proper font size calibration (Optional but recommended)
- `--output`: Filepath to output `.pdf` or `.json` report.
- `--verbose`: Enable detailed logging.
