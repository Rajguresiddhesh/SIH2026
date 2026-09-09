# Models

| file | model | role | trains on |
|---|---|---|---|
| `florence2_finetune.py` / `florence2_infer.py` | **Florence-2-base** | flagship — joint region detection + region OCR of declarations | one GPU, LoRA optional |
| `yolo_detect.py` | YOLOv8 / RT-DETR | detection-only baseline | one GPU |
| `recognizer.py` | PARSeq / TrOCR / Tesseract | text recogniser for the detect→OCR baseline (auto-selects) | pretrained |
| `layoutlmv3_fields.py` | LayoutLMv3-base | token-classification baseline (OCR words + 2-D layout → BIO tags) | one GPU |
| `donut_finetune.py` | Donut-base | OCR-free image → structured JSON baseline | one GPU |
| `distill.py` | teacher→student | pseudo-label pool → train YOLOv8-n student → INT8 TFLite for the Flutter app (§5) | one GPU |

All produce (directly or via `ml.inference.visual_extractor`) a validated
`ml.data.schema.ImageAnnotation`, so the eval harness and the compliance
pipeline treat them interchangeably.

## Checkpoint discovery

`ml.inference.VisualDeclarationExtractor` picks, in order:

1. `$PCR_FLORENCE_DIR` or `runs/florence2/`
2. `$PCR_YOLO_WEIGHTS` or `runs/detect/train/weights/best.pt` (+ Tesseract for values)
3. nothing → the app's LLM / regex-OCR paths handle the request

## Reporting (paper §6)

`ml/eval/run_all.py` benchmarks whichever checkpoint is active:
detection mAP, field P/R/F1 (exact + CER-fuzzy), end-to-end per-rule compliance
F1 / coverage, and — after `--calibrate` on the `geom` split — conformal
coverage vs target α. Swap the checkpoint env var and re-run for each baseline
row.
