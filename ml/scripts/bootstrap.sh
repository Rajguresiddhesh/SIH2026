#!/usr/bin/env bash
# End-to-end sanity run with synthetic data — no GPU, no labelling.
# Proves the data -> geometry -> rulebook -> conformal -> eval chain works.
set -euo pipefail
cd "$(dirname "$0")/../.."

PY="${PY:-python}"
DATA="${DATA:-data/pcr_synth}"

echo "== 1. generate synthetic PCR-Label =="
$PY -m ml.data.synth --out "$DATA" --n 400 --geom-frac 0.4 --seed 0

echo "== 2. unit tests =="
$PY -m ml.tests.test_ml

echo "== 3. calibrate conformal thresholds on the geom split =="
$PY -m ml.eval.run_all --root "$DATA" --calibrate --alpha 0.1

echo "== 4. evaluate (oracle predictor = upper bound / plumbing check) =="
$PY -m ml.eval.run_all --root "$DATA" --split geom --oracle --report out/synth_results

echo
echo "Done. Now collect real photos and:"
echo "  $PY -m ml.data.annotate_gemini --images data/raw --out data/pcr_label --split silver"
echo "  $PY -m ml.data.convert --root data/pcr_label --fmt florence --out data/florence"
echo "  accelerate launch -m ml.models.florence2_finetune --data data/florence --images data/pcr_label"
