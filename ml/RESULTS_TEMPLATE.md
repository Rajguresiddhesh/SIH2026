# Results — table skeletons (paper §6)

Fill from `python -m ml.eval.run_all --root data/pcr_label --split gold --report out/<model>`
and `--split geom` / `--split robust`. Swap `$PCR_FLORENCE_DIR` / `$PCR_YOLO_WEIGHTS`
between rows.

## Table 1 — Declaration region detection (`gold`)

| Model | mAP@[.5:.95] | mAP@.5 | AP `mrp` | AP `net_quantity` | AP `mfg_date` | AP `fssai` | AP `veg/nonveg` | AP `barcode` |
|---|---|---|---|---|---|---|---|---|
| YOLOv8-m | | | | | | | | |
| RT-DETR | | | | | | | | |
| Florence-2 (ours) | | | | | | | | |

## Table 2 — Field extraction (`gold`)

| Model | macro-F1 (fuzzy) | mean CER | F1 `mrp` | F1 `net_quantity` | F1 `manufacturer_address` | F1 `consumer_care` | F1 `mfg_date` |
|---|---|---|---|---|---|---|---|
| Tesseract + regex (v1) | | | | | | | |
| YOLOv8 + PARSeq + regex | | | | | | | |
| LayoutLMv3 (token-cls) | | | | | | | |
| Donut | | | | | | | |
| Gemini (zero-shot) | | | | | | | |
| Florence-2 (ours) | | | | | | | |

## Table 3 — Metric geometry (`geom`)

| Method | mm/px MAE | cap-height MAE (mm) | within Rule-9 tol. (%) | Rule 9 F1 | Rule 8 F1 |
|---|---|---|---|---|---|
| barcode prior only | | | | | |
| + panel rectify | | | | | |
| + known dimension | | | | | |
| ruler GT (upper bound) | 0.00 | 0.00 | 100 | | |

## Table 4 — End-to-end compliance (`gold`)

| System | macro-F1 (active) | selective acc. | coverage | crit-violation recall | ECE |
|---|---|---|---|---|---|
| v1 (regex + rulebook + EBM) | | | | | |
| Gemini vision | | | | | |
| ours (visual + geometry) | | | | | |
| ours + conformal (α=0.1) | | | | | |

## Table 5 — Conformal guarantee (`geom` test)

| Rule | α | empirical FP rate | empirical FN rate | coverage |
|---|---|---|---|---|
| R07_FONT | 0.10 | | | |
| R08_CLEAR | 0.10 | | | |

## Table 6 — Ablations

| Setting | field macro-F1 | compliance macro-F1 |
|---|---|---|
| silver only | | |
| silver + gold | | |
| + active-learning budget (vs random) | | |
| Florence-2 vs LayoutLMv3 vs Donut | | |

## Table 7 — On-device student (paper §5)

| Model | field macro-F1 | mAP | size (MB) | latency (ms, Pixel-class) |
|---|---|---|---|---|
| Florence-2 teacher | | | | |
| YOLOv8-n student (FP16 TFLite) | | | | |
| YOLOv8-n student (INT8 TFLite) | | | | |

## Table 8 — Robustness (`robust`)

| Perturbation | Δ field macro-F1 | Δ compliance macro-F1 |
|---|---|---|
| perspective | | |
| motion blur | | |
| glare | | |
| low-light | | |
| JPEG q<40 | | |
