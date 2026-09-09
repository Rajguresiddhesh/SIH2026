# Dataset card — PCR-Label

## Summary

Images of Indian pre-packaged retail commodities annotated for **Legal
Metrology (Packaged Commodities) Rules, 2011** compliance: 17 statutory
declaration regions with transcribed values, and a ruler-calibrated subset with
ground-truth millimetre text geometry.

| split | images (target) | labels | use |
|---|---|---|---|
| `silver` | 3,000–5,000 | Gemini grounded auto-annotation | pre-training / fine-tuning |
| `gold` | 200–500 | human-verified regions + values | fine-tuning + primary eval |
| `geom` | 80–150 | + mm ruler → mm/px + glyph cap/x-heights | Rule 7/8/9 evaluation |

## Collection

- **Sources:** first-party photographs of products bought at retail; publicly
  listed e-commerce product images (Open Food Facts image dump; retailer product
  pages). No scraping of paywalled or login-gated content.
- **Diversity targets:** ≥8 commodity categories (staples, snacks, beverages,
  dairy, personal care, nutraceutical, household, spices); pouch / bottle / can /
  carton / blister; front & back views; a robustness slice with perspective,
  glare, low-light, blur, and Devanagari-dominant labels.

## Annotation

- Silver: `ml/data/annotate_gemini.py` (Gemini 2.x, grounded prompt, schema-
  validated). Inherits the teacher's biases — **not** used for evaluation.
- Gold / geom: human, per `ml/ANNOTATION_GUIDE.md`, tool-agnostic (labelme /
  CVAT export mapped to the schema). Active-learning selection
  (`ml/data/active_learning.py`) so the budget targets informative images.
- Inter-annotator agreement: double-annotate 10 % of `gold`; report mean IoU and
  value-CER; adjudicate disagreements.

## Splits

`silver` is randomly partitioned train/val/test (80/10/10, seed 42) within
`ml/data/split.py`. `gold` and `geom` are held out for evaluation only; a
portion of `gold` may be folded into fine-tuning (reported as an ablation).

## Ethics, licensing, PII

- Brand marks and trade dress are incidental to the compliance task; no
  endorsement or disparagement is implied.
- Consumer-care blocks contain **company** contact details (public on the pack),
  not personal data. No faces, no individuals.
- Released annotations: CC BY-NC 4.0. Images: only first-party captures are
  redistributed; for third-party images we release URLs + annotation offsets.
- The dataset is a screening aid for research, **not** a certified compliance
  determination.

## Known limitations

- Single-country ruleset (India, PCR 2011 as amended).
- Silver labels reflect one LLM's reading; systematic teacher errors (e.g. on
  stylised Devanagari) propagate to pre-training.
- `geom` assumes the ruler lies in the panel plane; out-of-plane error is not
  modelled.
- Class imbalance: `mrp_tax_clause`, `country_of_origin`, `packer_importer`,
  `nonveg_mark` are rare — per-class metrics reported, not just macro.

## Maintenance

Versioned in `data/pcr_label/index.json` (`version` field). Additive updates
bump the minor version; schema changes bump the major and ship a migration.
