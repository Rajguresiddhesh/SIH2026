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

If a Gemini key *is* present it only *fills gaps* in the registry data; the
verdict is always the deterministic rulebook's.

## Installation

Requires **Python 3.10+**.

```bash
# from the repository root
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # macOS / Linux
pip install -r requirements.txt
```

The first run downloads the EasyOCR models (~200 MB) and, if no trained model is
present, trains a small EBM on synthetic data — allow ~30–60 s on the first
launch only.

### Configuration (optional)

Create a `.env` file in the repository root:

```ini
# Enables the Gemini multimodal-vision path and LLM gap-filling for bar codes.
GEMINI_API_KEY=your-google-ai-studio-key

# Authoritative Indian GTIN data (licensed GS1 India / DataKart subscription).
# Without these the free registries (Open Food Facts, UPCItemDB, Wikidata) are used.
GS1_INDIA_API_KEY=your-datakart-key
GS1_INDIA_API_URL=https://datakartapigateway.gs1india.org/api/v1/product/gtin/{gtin}
```

The Gemini key can also be set at runtime from the web UI (🔑 button, top-right),
which writes it back to `.env`.

## Running the web app

```bash
python app.py
```

Then open <http://localhost:5000>. Three input modes:

| Mode | What it does |
|---|---|
| **Upload** | Drag-and-drop front / back / ruler label photos. |
| **Camera** | Capture the label live via the browser camera. |
| **Scan Barcode** | Point the camera at a bar code, upload a bar-code photo, or type a GTIN. The scan card shows a live GTIN verification checklist and the product resolved from the registries; **"Verify Every Declaration (Rules 2011)"** runs the full audit straight from the bar code — no photo needed. |

### HTTP API

- `POST /analyze` — `multipart/form-data` (`front`, `back`, `ruler`,
  `barcode`/`barcode_number`, `package_height_mm`) **or** JSON
  (`front_b64`, `back_b64`, `ruler_b64`, `barcode_b64`, `barcode_number`,
  `package_height_mm`). Optional `X-Gemini-Api-Key` header. Returns the
  compliance score, per-rule results, product identification and a report id.
- `GET /download/<report_id>` — the generated PDF.
- `POST /api/scan-barcode` — decode + structurally verify a bar code. Body:
  an image (`multipart` file or `image_b64`) or `barcode_number`. Add
  `?lookup=1` (or `"lookup": true`) to also resolve the product in the
  registries.
- `GET/POST /api/settings/api-key` — read / store the Gemini key.

## Command line

```bash
# Image audit
python -m legal_metrology_ml.main --front front.jpg --back back.jpg --output report.pdf

# Bar-code-only audit (deterministic; no key required)
python -c "from legal_metrology_ml.main import run_barcode_pipeline; \
r = run_barcode_pipeline('8901030928239', output_path='report.pdf'); \
print(r.compliance_score.final_score, r.compliance_score.star_label)"
```

### `main.py` options

- `--front` — path to the FRONT label image (required for the image pipeline).
- `--back` — path to the BACK / SIDE label image (recommended).
- `--ruler` — image showing a mm ruler for font-size calibration (Rule 9).
- `--package-height-mm` — real-world package height in mm (alternative
  calibration).
- `--output` — output report path (`.pdf` or `.json`).
- `--verbose` — detailed logging.
