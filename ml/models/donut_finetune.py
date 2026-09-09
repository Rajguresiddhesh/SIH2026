"""Donut (OCR-free) baseline: image -> structured declaration JSON.

Fine-tunes `naver-clova-ix/donut-base` to emit the `gt_parse` produced by
`ml/data/convert.py --fmt donut`.

    python -m ml.models.donut_finetune --data data/donut --images data/pcr_label --epochs 30
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--model", default="naver-clova-ix/donut-base")
    ap.add_argument("--out", type=Path, default=Path("runs/donut"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=768)
    a = ap.parse_args()

    import torch
    from PIL import Image
    from torch.utils.data import Dataset
    from transformers import (
        DonutProcessor,
        Trainer,
        TrainingArguments,
        VisionEncoderDecoderModel,
    )

    proc = DonutProcessor.from_pretrained(a.model)
    model = VisionEncoderDecoderModel.from_pretrained(a.model)

    new_tokens = ["<s_pcr>", "</s_pcr>"] + [f"<s_{k}>" for k in _KEYS] + [f"</s_{k}>" for k in _KEYS]
    proc.tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
    model.decoder.resize_token_embeddings(len(proc.tokenizer))
    model.config.decoder_start_token_id = proc.tokenizer.convert_tokens_to_ids("<s_pcr>")
    model.config.pad_token_id = proc.tokenizer.pad_token_id

    rows = []
    for sf in ("train.jsonl", "silver.jsonl", "gold.jsonl"):
        p = a.data / sf
        if p.exists():
            rows += [json.loads(x) for x in p.read_text().splitlines() if x.strip()]

    class DS(Dataset):
        def __len__(self): return len(rows)

        def __getitem__(self, i):
            r = rows[i]
            img = Image.open(a.images / r["image"]).convert("RGB")
            pixel_values = proc(img, return_tensors="pt").pixel_values.squeeze(0)
            target = "<s_pcr>" + _to_tags(json.loads(r["ground_truth"])["gt_parse"]) + "</s_pcr>"
            labels = proc.tokenizer(target, add_special_tokens=False, max_length=a.max_len,
                                    padding="max_length", truncation=True,
                                    return_tensors="pt").input_ids.squeeze(0)
            labels[labels == proc.tokenizer.pad_token_id] = -100
            return {"pixel_values": pixel_values, "labels": labels}

    Trainer(
        model=model,
        args=TrainingArguments(output_dir=str(a.out), num_train_epochs=a.epochs,
                               per_device_train_batch_size=a.bs, learning_rate=3e-5,
                               warmup_ratio=0.05, save_strategy="epoch", save_total_limit=2,
                               fp16=torch.cuda.is_available(), report_to=[]),
        train_dataset=DS(),
    ).train()
    model.save_pretrained(a.out)
    proc.save_pretrained(a.out)


_KEYS = [
    "generic_name", "net_quantity", "mrp", "mrp_tax_clause", "mfg_date", "expiry_date",
    "manufacturer_name", "manufacturer_address", "packer_importer", "consumer_care",
    "country_of_origin", "fssai", "barcode", "veg_mark", "nonveg_mark",
    "principal_display_panel", "package_type", "view",
]


def _to_tags(d: dict) -> str:
    parts = []
    for k in _KEYS:
        if k not in d:
            continue
        v = d[k]
        vals = v if isinstance(v, list) else [v]
        for item in vals:
            parts.append(f"<s_{k}>{'' if item is True else item}</s_{k}>")
    return "".join(parts)


if __name__ == "__main__":
    main()
