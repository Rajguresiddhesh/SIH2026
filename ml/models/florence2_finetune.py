"""Fine-tune Florence-2 for grounded declaration extraction on PCR-Label.

Two task formulations, trained jointly from the converter output
(`ml/data/convert.py --fmt florence`):

  <OD>               -> "<class><loc_x0><loc_y0><loc_x1><loc_y1> ..."
  <OCR_WITH_REGION>  -> "<value><loc_x0><loc_y0><loc_x1><loc_y1> ..."

Run (single GPU / Colab):

    accelerate launch -m ml.models.florence2_finetune \
        --data data/florence --images data/pcr_label \
        --model microsoft/Florence-2-base-ft --epochs 8 --bs 4 --lora

The checkpoint drops into `runs/florence2/` and is picked up automatically by
`ml.inference.visual_extractor`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# heavy deps imported lazily inside main() so --help works without them


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def build_dataset(data_dir: Path, images_dir: Path, split_files: list[str]):
    from PIL import Image
    from torch.utils.data import Dataset

    rows: list[dict] = []
    for sf in split_files:
        rows.extend(_read_jsonl(data_dir / sf))

    class FlorenceDS(Dataset):
        def __len__(self) -> int:
            return len(rows)

        def __getitem__(self, i: int):
            r = rows[i]
            img = Image.open(images_dir / r["image"]).convert("RGB")
            return {"image": img, "prefix": r["prefix"], "suffix": r["suffix"]}

    return FlorenceDS()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True, help="dir with {train,val,gold}.jsonl")
    ap.add_argument("--images", type=Path, required=True, help="PCR-Label root (has images/)")
    ap.add_argument("--model", default="microsoft/Florence-2-base-ft")
    ap.add_argument("--out", type=Path, default=Path("runs/florence2"))
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--lora", action="store_true")
    ap.add_argument("--freeze-vision", action="store_true")
    a = ap.parse_args()

    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoProcessor,
        Trainer,
        TrainingArguments,
    )

    processor = AutoProcessor.from_pretrained(a.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, trust_remote_code=True, torch_dtype=torch.float16
    )

    if a.freeze_vision:
        for p in model.vision_tower.parameters():
            p.requires_grad_(False)
    if a.lora:
        from peft import LoraConfig, get_peft_model

        model = get_peft_model(
            model,
            LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                       target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                       "linear", "Conv2d", "lm_head", "fc1", "fc2"]),
        )
        model.print_trainable_parameters()

    train_ds = build_dataset(a.data, a.images, ["train.jsonl", "silver.jsonl"])
    eval_ds = build_dataset(a.data, a.images, ["val.jsonl", "gold.jsonl"])

    def collate(batch: list[dict]):
        prompts = [b["prefix"] for b in batch]
        images = [b["image"] for b in batch]
        answers = [b["suffix"] for b in batch]
        inputs = processor(text=prompts, images=images, return_tensors="pt",
                           padding=True, truncation=True)
        labels = processor.tokenizer(text=answers, return_tensors="pt",
                                     padding=True, truncation=True,
                                     max_length=1024).input_ids
        labels[labels == processor.tokenizer.pad_token_id] = -100
        inputs["labels"] = labels
        return inputs

    args = TrainingArguments(
        output_dir=str(a.out),
        num_train_epochs=a.epochs,
        per_device_train_batch_size=a.bs,
        per_device_eval_batch_size=a.bs,
        gradient_accumulation_steps=max(1, 16 // a.bs),
        learning_rate=a.lr,
        warmup_ratio=0.05,
        weight_decay=0.01,
        logging_steps=25,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        dataloader_num_workers=4,
        remove_unused_columns=False,
        report_to=[],
    )
    Trainer(
        model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds,
        data_collator=collate,
    ).train()

    a.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(a.out)
    processor.save_pretrained(a.out)
    (a.out / "MODEL_CARD.md").write_text(
        f"# Florence-2 PCR-Label\nbase: {a.model}\nepochs: {a.epochs}\nlora: {a.lora}\n"
    )
    print(f"saved -> {a.out}")


if __name__ == "__main__":
    main()
