PY ?= python
DATA ?= data/pcr_label

.PHONY: help app test ml-test synth bootstrap calibrate eval convert-florence lint

help:
	@echo "app              run the Flask compliance server (localhost:5000)"
	@echo "test             backend OCR smoke test (test_ocr.py)"
	@echo "ml-test          ml/ unit suite (10 tests, no GPU)"
	@echo "bootstrap        synthetic end-to-end sanity run (no GPU, no labels)"
	@echo "synth            generate a synthetic PCR-Label set into data/pcr_synth"
	@echo "calibrate        fit conformal thresholds on the geom split -> runs/conformal.json"
	@echo "eval SPLIT=gold  run the full eval harness (needs a trained checkpoint or --oracle)"
	@echo "convert-florence PCR-Label -> Florence-2 jsonl"

app:
	$(PY) app.py

test:
	$(PY) test_ocr.py zanskar_front.jpg tesseract

ml-test:
	$(PY) -m ml.tests.test_ml

bootstrap:
	bash ml/scripts/bootstrap.sh

synth:
	$(PY) -m ml.data.synth --out data/pcr_synth --n 500 --geom-frac 0.35 --seed 0

calibrate:
	$(PY) -m ml.eval.run_all --root $(DATA) --calibrate --alpha 0.1

SPLIT ?= gold
eval:
	$(PY) -m ml.eval.run_all --root $(DATA) --split $(SPLIT) --report out/$(SPLIT)

convert-florence:
	$(PY) -m ml.data.convert --root $(DATA) --fmt florence --out data/florence

lint:
	$(PY) -m ruff check ml legal_metrology_ml || true
