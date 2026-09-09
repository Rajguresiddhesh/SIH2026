# `ml/` — PCR-Label: visual document understanding + metric text geometry

Research code for [`PAPER.md`](../PAPER.md). Replaces the two weakest links of
the v1 system — regex field parsing and the heuristic symbol detector — with a
trained grounded-extraction model, and adds a metric-scale geometry pipeline so
the Rule 7/8/9 font/space checks are **measured** instead of skipped.

```
ml/
  data/        taxonomy (17 classes) · annotation schema · Gemini auto-annotator ·
               format converters · active-learning sampler · split manager
  models/      Florence-2 (flagship) · YOLOv8 detector · LayoutLMv3 · training + infer
  geometry/    mm/px recovery (ruler / dimension / reference / barcode) · panel
               rectification · per-glyph cap/x-height · uncertainty propagation
  eval/        detection mAP · field P/R/F1 (exact + CER) · per-rule compliance
               F1/coverage · split-conformal PASS/FAIL/ABSTAIN
  inference/   VisualDeclarationExtractor · annotation→PackageData · run_ml_pipeline
```

## Install

```bash
python -m venv .venv-ml && source .venv-ml/bin/activate    # CUDA box
pip install -r ml/requirements.txt
pip install -r requirements.txt          # the base app, for the rulebook
```

## Reproduce the pipeline

### 1. Collect + auto-annotate (silver)

```bash
# put raw product photos under data/raw/
python -m ml.data.annotate_gemini --images data/raw --out data/pcr_label --split silver
python -m ml.data.split --root data/pcr_label --auto 0.8 0.1 0.1
```

### 2. Human gold set via active learning

```bash
python -m ml.data.convert --root data/pcr_label --fmt florence --out data/florence
python -m ml.models.florence2_finetune --data data/florence --images data/pcr_label --epochs 4 --lora   # quick bootstrap
python -m ml.models.florence2_infer ...            # write pool predictions -> runs/florence2/pool_preds.jsonl
python -m ml.data.active_learning --root data/pcr_label --preds runs/florence2/pool_preds.jsonl --budget 300 --out queue.txt
#  ... hand-correct the 300 images (labelme / CVAT, export back to the schema) ...
python -m ml.data.split --root data/pcr_label --promote queue.txt --to gold
```

### 3. Geometry ground truth

Photograph 80–150 products with a mm ruler in frame; mark the ruler endpoints
and a few glyph baselines. Set `split: geom` and fill `geometry:` in the
annotation JSON (`ml/data/schema.py::GeometryGT`).

### 4. Train the models

```bash
python -m ml.data.convert --root data/pcr_label --fmt florence   --out data/florence
python -m ml.models.florence2_finetune --data data/florence --images data/pcr_label --epochs 8 --lora

# baselines
python -m ml.models.yolo_detect train --root data/pcr_label --model yolov8m.pt --epochs 120
python -m ml.data.convert --root data/pcr_label --fmt layoutlmv3 --out data/llmv3
python -m ml.models.layoutlmv3_fields prepare --root data/pcr_label --out data/llmv3
python -m ml.models.layoutlmv3_fields train --data data/llmv3 --images data/pcr_label
```

### 5. Calibrate conformal thresholds

```bash
python -m ml.eval.run_all --root data/pcr_label --calibrate   # writes runs/conformal.json
```

### 6. Evaluate (regenerates every table/figure)

```bash
python -m ml.eval.run_all --root data/pcr_label --report out/results
```

## Use the trained pipeline in the app

Point the app at the checkpoint and select the engine:

```bash
export PCR_FLORENCE_DIR=runs/florence2          # or PCR_YOLO_WEIGHTS=...
python app.py
curl -X POST 'http://localhost:5000/analyze?engine=ml' -F front=@label.jpg -F back=@back.jpg
```

`legal_metrology_ml.main.run_pipeline(..., engine="ml")` also works from Python
and the CLI. With no checkpoint present, `engine="auto"` silently falls back to
the LLM / local-OCR paths.

## On-device (future work)

`tools/` (in the `ML-SIH-` repo) distils the Florence-2 extractor to an INT8
TFLite student for the Flutter app — §5 of the paper.
