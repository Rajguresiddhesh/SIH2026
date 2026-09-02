# Legal Metrology ML Pipeline

A comprehensive machine learning and rule-based system for evaluating compliance of packaged products according to Legal Metrology standards.

## Architecture

This pipeline is organized into 5 layers:

1. **Layer 1: Feature Extraction**: Processes raw images to extract text (OCR), detect standard symbols, segment panels, and estimate real-world font metrics based on dimension calibration.
2. **Layer 2: Data Normalization**: Takes extracted raw inputs and maps them onto a standardized, well-typed schema (`NormalizedPackageData`).
3. **Layer 3: ML Model**: Uses an Explainable Boosting Machine (EBM) model to score the probability of compliance from a holistic perspective.
4. **Layer 4: Rulebook Engine**: Performs deterministic, rule-based checks applying legal metrology regulations (e.g., minimum font size tables, mandatory fields).
5. **Layer 5: Aggregation & Reporting**: Combines ML predictions with rulebook diffs to generate a final `ComplianceScore`, along with actionable recommendations. Outputs JSON or PDF reports.

## Bar-code compliance audit (Rule 6, LM(PC) Rules 2011)

Scanning a bar code now runs a **deterministic** audit — no Gemini key needed:

1. **`gs1.classify_gtin`** structurally verifies the number: GTIN-8/12/13/14
   length, GS1 mod-10 check digit, issuing GS1 member organisation, GS1 India
   (prefix `890`), and reserved/restricted ranges (retailer-internal, coupons,
   ISBN/ISSN book-land).
2. **`data_sources.lookup_product`** resolves the GTIN to product declarations
   from, in order of trust:
   - GS1 India / DataKart (licensed — set `GS1_INDIA_API_KEY`, optionally
     `GS1_INDIA_API_URL`)
   - Open Food Facts + Open Beauty / Products / Pet Food Facts (free)
   - UPCItemDB trial endpoint (free, rate-limited)
   - Wikidata (`P3962`, free)
3. The record is normalised to `PackageData` and run through the **full
   rulebook** (`layer4_rulebook_engine`) plus bar-code-specific checks
   (`barcode_rules`: `B01`–`B06`).
4. Declarations the registry cannot supply are reported **INCONCLUSIVE**
   ("verify on the physical label"), and their weight counts against the score
   so an unidentified product never reads as "100% compliant".

If a Gemini key *is* present it only *fills gaps* in the registry data; the
verdict is always the deterministic rulebook's.

CLI: `python -m legal_metrology_ml.main --front label.jpg` still works; the web
app exposes it at `POST /analyze` (with `barcode_number` / a bar-code image) and
`POST /api/scan-barcode?lookup=1`.

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
