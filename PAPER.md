# Automated Legal-Metrology Compliance of Pre-Packaged Commodities via Visually-Rich Document Understanding and Metric-Scale Text Geometry

Working title / thesis outline. Code for every claim lives in [`ml/`](ml/).

---

## 1. Problem

The Legal Metrology (Packaged Commodities) Rules, 2011 (India) require a fixed
set of declarations on every retail package — name & address of the
manufacturer/packer/importer, common name, **net quantity in standard units**,
month & year of manufacture, **retail sale price "inclusive of all taxes"**,
consumer-care details, country of origin (imported), and — critically —
**minimum heights for letters and numerals** that scale with the area of the
principal display panel (Rule 9), plus clear space around the quantity
declaration (Rule 8).

Manual field inspection does not scale to e-commerce catalogues or high-volume
retail. Prior automation (this repository's v1, and comparable systems) reduces
to: OCR → regex field parsing → a hand-written rulebook, with an LLM used
zero-shot. Two failure modes dominate:

1. **Extraction** — regex parsing over noisy OCR misses fields on stylised,
   multi-script (Devanagari + Latin), curved-surface labels.
2. **Geometry** — Rules 7–9 need *real-world millimetre* measurements. Without a
   scale reference the system cannot judge them and abstains on the very rules
   that inspectors cite most.

## 2. Contributions

1. **PCR-Label** — a benchmark of Indian pre-packaged-commodity images with
   (a) declaration-region polygons for 17 classes, (b) transcribed field
   values, (c) a **ruler-calibrated geometry subset** giving ground-truth
   mm/px and per-glyph cap/x-heights. Built with an **LLM-assisted, active-
   learning** annotation pipeline (§4).
2. **A grounded declaration-extraction model** — Florence-2 fine-tuned for
   joint region detection + region OCR of statutory declarations, benchmarked
   against YOLOv8+PARSeq+regex, LayoutLMv3 token-classification, Donut, and
   zero-shot Gemini.
3. **Metric-scale text geometry** — a pipeline that (a) recovers absolute
   scale from the GS1 barcode as a standardised reference under homography,
   (b) rectifies the principal display panel (incl. cylinder unwrapping for
   cans/bottles), (c) estimates per-glyph cap-height / x-height / stroke width,
   and (d) propagates measurement uncertainty into the Rule 7–9 decision.
4. **Calibrated, abstaining compliance** — split-conformal prediction per rule,
   turning the ad-hoc "INCONCLUSIVE" label into an abstention with a
   distribution-free coverage guarantee; reported with reliability diagrams
   and selective-risk curves.
5. **Teacher→student distillation** — the fine-tuned extractor distilled to an
   INT8 TFLite model for fully-offline on-device inference (integrated with the
   Flutter app; see the `ML-SIH-` repo).

## 3. Related work

- **Visually-rich document understanding:** LayoutLM/v2/v3, DocFormer, ERNIE-
  Layout (token classification with 2-D positions); Donut, Pix2Struct,
  Dessurt (OCR-free image→sequence); Florence-2 (unified detection + region
  captioning + OCR).
- **Document layout analysis / scene text:** PubLayNet, DocLayNet; DBNet,
  CRAFT (detection); PARSeq, TrOCR, ABINet (recognition).
- **Metric scale from a single image:** Depth-Anything-V2 (metric),
  UniDepth, Metric3D; reference-object photogrammetry; GS1 barcode geometry.
- **Selective prediction / uncertainty:** split-conformal prediction
  (Vovk; Angelopoulos & Bates), selective classification (Geifman & El-Yaniv),
  deep ensembles, temperature scaling.
- **Regulatory-text / label compliance automation:** rule-based label checkers,
  LLM-as-inspector prompting, nutrition-label extraction datasets.

## 4. Dataset — PCR-Label

| Split | Images | Labels | Purpose |
|---|---|---|---|
| `silver` | ~3–5k | Gemini auto-annotation (§4.1) | pre-training / fine-tuning |
| `gold` | 200–500 | human-verified regions + values | fine-tuning + primary eval |
| `geom`  | 80–150 | + mm ruler in frame → true mm/px, glyph heights | Rule 7–9 evaluation |

**4.1 LLM-assisted annotation.** `ml/data/annotate_gemini.py` prompts Gemini
with a grounded schema (normalised `[x0,y0,x1,y1]` + value + script + a
confidence per field). Output is validated against `ml/data/schema.py`.

**4.2 Active learning.** `ml/data/active_learning.py` ranks un-labelled images
by (i) model predictive entropy, (ii) disagreement between the model and the
Gemini annotation, (iii) rareness of the predicted declaration set — so the
200–500 human hours land on the most informative images.

**4.3 Geometry ground truth.** For the `geom` split the annotator marks the
ruler's two endpoints (→ mm/px) and, for a sampled set of glyphs, the baseline
/ cap-line / mean-line (→ true cap-height, x-height in mm).

**4.4 Ethics / licensing.** Only publicly-listed retail product photos and
first-party captures; brand marks are incidental. Dataset card:
`ml/DATASET_CARD.md`.

## 5. Method

### 5.1 Declaration extraction
Fine-tune **Florence-2-base** with the `<OD>` + `<OCR_WITH_REGION>` task
formulation over the 17-class taxonomy (`ml/data/taxonomy.py`).
`ml/models/florence2_finetune.py`. Baselines in `ml/models/`.

Output → `PackageData` via `ml/inference/visual_extractor.py`, which slots into
Layer 1/2 of `legal_metrology_ml`, replacing the regex `TextParser` and the
heuristic `ObjectDetector` when weights are present (graceful fallback kept).

### 5.2 Metric text geometry
`ml/geometry/`:
- `calibration.py` — barcode-reference scale + panel homography; optional
  metric-depth fallback.
- `rectify.py` — planar/cylindrical unwrapping of the display panel.
- `glyph_metrics.py` — per-glyph cap/x-height + stroke width with a
  perspective correction; uncertainty from calibration + detector variance.
- Feeds `mrp_font_height_mm`, `net_qty_font_height_mm`,
  `min_font_height_mm`, panel area, clear-space fields.

### 5.3 Compliance decision
Existing 38-rule engine, now fed real measurements. Wrapped by
`ml/eval/conformal.py` (split-conformal, per rule): predict PASS/FAIL only
inside the calibrated set, else ABSTAIN(→INCONCLUSIVE), at a user-set risk α.

## 6. Experiments

- **E1 Detection:** mAP@[.5:.95], per-class AP, on `gold`.
- **E2 Field extraction:** precision/recall/F1 per declaration (exact + CER-
  normalised fuzzy match); end-to-end vs pipeline.
- **E3 Geometry:** mm/px MAE, cap-height MAE (mm) and % within Rule-9
  tolerance, on `geom`; ablate barcode-only vs +rectify vs +depth.
- **E4 Compliance:** per-rule F1 and PR-AUC vs the v1 regex+rulebook, vs
  zero-shot Gemini, vs the fine-tuned pipeline; calibration (ECE, reliability
  diagram); conformal coverage vs target; selective-risk / coverage curve.
- **E5 Ablations:** silver-only vs +gold; active vs random labelling budget;
  Florence-2 vs LayoutLMv3 vs Donut; distilled TFLite vs full model
  (accuracy / latency / size).
- **E6 Robustness:** perspective, blur, glare, low-light, script mix,
  packaging type (pouch / bottle / can / carton).

## 7. Limitations

Single-country ruleset; GS1-reference scale assumes a printed barcode at
nominal magnification; conformal guarantees are marginal (not per-brand);
LLM-assisted silver labels inherit Gemini's biases (mitigated by the gold set).

## 8. Reproducibility

Everything is scripted under `ml/` with pinned `ml/requirements.txt`, fixed
seeds, and a `ml/eval/run_all.py` that regenerates every table/figure from a
released checkpoint + the dataset.
