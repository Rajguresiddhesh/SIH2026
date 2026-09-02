"""Test script for OCR Engine using Tesseract.

Extracts all text written on the target image label, prints bounding boxes,
confidence scores, full reconstructed text, and parsed Legal Metrology fields.
"""
from __future__ import annotations

import sys
from pathlib import Path

from legal_metrology_ml.layer1_feature_extraction.ocr_engine import OCREngine
from legal_metrology_ml.layer1_feature_extraction.text_parser import TextParser


def run_ocr(image_path: str, engine_name: str = "tesseract") -> None:
    path = Path(image_path)
    if not path.exists():
        print(f"Error: Image not found at '{image_path}'")
        return

    print("=" * 70)
    print(f"Running OCR ({engine_name}) on: {path.name}")
    print("=" * 70)

    engine = OCREngine(engine=engine_name)
    results = engine.extract_text(str(path))

    print(f"\nTotal text blocks detected: {len(results)}\n")
    print("-" * 70)
    print(f"{'No.':<4} | {'Conf':<6} | {'Coordinates (x, y, w, h)':<25} | {'Text'}")
    print("-" * 70)

    for i, r in enumerate(results, start=1):
        x = int(r.bbox[0][0])
        y = int(r.bbox[0][1])
        w = int(r.width_px)
        h = int(r.height_px)
        coord_str = f"({x}, {y}, {w}x{h})"
        print(f"{i:<4} | {r.confidence * 100:>5.1f}% | {coord_str:<25} | {r.text}")

    print("-" * 70)
    print("\n" + "=" * 70)
    print("FULL EXTRACTED TEXT:")
    print("=" * 70)
    full_text = engine.extract_full_text(str(path))
    print(full_text)
    print("=" * 70)

    # Run TextParser on extracted results
    parser = TextParser()
    parsed = parser.parse(results)

    print("\nPARSED LEGAL METROLOGY DECLARATIONS:")
    print("-" * 50)
    print(f"Commodity Name:       {parsed.commodity_name}")
    print(f"MRP:                  {parsed.mrp_value} (raw: {parsed.mrp_raw_text})")
    print(f"Net Quantity:         {parsed.net_quantity_value} {parsed.net_quantity_unit}")
    print(f"FSSAI License:        {parsed.fssai_license}")
    print(f"Mfg Date:             {parsed.manufacture_date}")
    print(f"Expiry Date:          {parsed.expiry_date}")
    print(f"Best Before:          {parsed.best_before}")
    print(f"Manufacturer Name:    {parsed.manufacturer_name}")
    print(f"Manufacturer Address: {parsed.manufacturer_address}")
    print(f"Packer Name:          {parsed.packer_name}")
    print(f"Packer Address:       {parsed.packer_address}")
    print(f"Importer Name:        {parsed.importer_name}")
    print(f"Consumer Care Phone:  {parsed.consumer_care_phone}")
    print(f"Consumer Care Email:  {parsed.consumer_care_email}")
    print(f"Consumer Care Addr:   {parsed.consumer_care_address}")
    print(f"Has English Text:     {parsed.has_english_text}")
    print(f"Has Hindi Text:       {parsed.has_hindi_text}")
    print("-" * 50)


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "product.jpeg"
    engine = sys.argv[2] if len(sys.argv) > 2 else "tesseract"
    run_ocr(target, engine)
