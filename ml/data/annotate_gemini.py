"""LLM-assisted grounded annotation for PCR-Label (silver labels).

Prompts Gemini for every statutory declaration region with a normalised
bounding box + transcribed value, validates the response against
``schema.ImageAnnotation``, and writes one JSON per image.

Usage
-----
    python -m ml.data.annotate_gemini --images data/raw --out data/pcr_label \
        --split silver [--limit 500] [--overwrite]

Requires ``GEMINI_API_KEY`` (env or ``.env``); reuses the project's existing
Gemini HTTP client.
"""
from __future__ import annotations

import argparse
import base64
import json
import logging
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from legal_metrology_ml.llm.llm_compliance_engine import LLMComplianceEngine, get_api_key  # noqa: E402

from ml.data.schema import BBox, ImageAnnotation, Region  # noqa: E402
from ml.data.taxonomy import CLASSES, TEXTUAL  # noqa: E402

logger = logging.getLogger("annotate_gemini")

_IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

_PROMPT = f"""
You are annotating a photo of an Indian pre-packaged retail commodity for a
computer-vision dataset. Find EVERY occurrence of the following statutory
declaration elements and return their location and text.

Classes (use these exact strings):
{json.dumps(CLASSES, indent=0)}

Rules:
- Coordinates are normalised to [0,1]: x0,y0 = top-left, x1,y1 = bottom-right,
  with x0<x1 and y0<y1. (0,0) is the top-left of the image.
- "principal_display_panel" = the single largest face/panel that carries the
  brand + net quantity; one box, the panel outline.
- "net_quantity" = the whole "Net Wt. 500 g" style block; "net_quantity_unit" =
  just the unit token ("g", "kg", "ml", "N") inside it.
- "mrp" = the price block (e.g. "MRP Rs. 45.00"); "mrp_tax_clause" = the words
  "inclusive of all taxes" / "incl. of all taxes" if present (may be separate).
- "veg_mark" = green filled circle in a green square; "nonveg_mark" = brown
  filled triangle in a brown square.
- "barcode" = the GS1 bar-code symbol (the bars, not the digits).
- Only include a class if it is actually visible. Do NOT invent boxes.
- For textual classes give the transcribed "value" verbatim (keep the original
  script; transliterate nothing). Set "script" to "latin", "devanagari" or
  "mixed".
- Give a per-region "confidence" in [0,1].

Return ONLY this JSON (no markdown fence):
{{
  "package_type": "pouch|bottle|can|carton|box|blister|unknown",
  "view": "front|back|side",
  "is_imported": true|false,
  "regions": [
    {{"cls": "mrp", "x0": 0.0, "y0": 0.0, "x1": 0.0, "y1": 0.0,
      "value": "MRP Rs. 45.00", "script": "latin", "confidence": 0.9}}
  ]
}}
"""


def _list_images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in _IMG_EXT)


def _image_part(path: Path) -> dict:
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return {"mime_type": mime, "data": base64.b64encode(path.read_bytes()).decode()}


def _image_size(path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as im:
        return im.width, im.height


def _coerce(raw: dict, image_rel: str, w: int, h: int, split: str) -> ImageAnnotation:
    regions: list[Region] = []
    for r in raw.get("regions", []) or []:
        cls = str(r.get("cls", "")).strip()
        if cls not in CLASSES:
            continue
        try:
            bbox = BBox(
                x0=_clamp(r.get("x0", 0.0)),
                y0=_clamp(r.get("y0", 0.0)),
                x1=_clamp(r.get("x1", 0.0)),
                y1=_clamp(r.get("y1", 0.0)),
            )
        except Exception:
            continue
        if bbox.area < 1e-5:
            continue
        val = r.get("value")
        regions.append(
            Region(
                cls=cls,
                bbox=bbox,
                value=(str(val).strip() if val and cls in TEXTUAL else None),
                script=r.get("script"),
                source="gemini",
                confidence=float(r.get("confidence", 0.5) or 0.5),
            )
        )
    return ImageAnnotation(
        image=image_rel,
        width=w,
        height=h,
        split=split,
        view=str(raw.get("view", "front")),
        package_type=str(raw.get("package_type", "unknown")),
        is_imported=bool(raw.get("is_imported", False)),
        regions=regions,
        annotator_notes="auto:gemini",
    )


def _clamp(v) -> float:
    try:
        return min(1.0, max(0.0, float(v)))
    except (TypeError, ValueError):
        return 0.0


def annotate_dir(
    images_root: Path,
    out_root: Path,
    split: str = "silver",
    limit: int | None = None,
    overwrite: bool = False,
    sleep_s: float = 0.5,
) -> None:
    key = get_api_key()
    if not key:
        raise SystemExit("GEMINI_API_KEY not set (env or .env).")
    engine = LLMComplianceEngine(api_key=key)

    (out_root / "images").mkdir(parents=True, exist_ok=True)
    (out_root / "annotations").mkdir(parents=True, exist_ok=True)

    imgs = _list_images(images_root)
    if limit:
        imgs = imgs[:limit]
    logger.info("Annotating %d images -> %s", len(imgs), out_root)

    ok = fail = 0
    for i, path in enumerate(imgs, 1):
        stem = path.stem
        ann_path = out_root / "annotations" / f"{stem}.json"
        if ann_path.exists() and not overwrite:
            continue
        try:
            w, h = _image_size(path)
            raw = engine._call_gemini_vision(_PROMPT, [_image_part(path)], key)  # noqa: SLF001
            ann = _coerce(raw, image_rel=f"images/{path.name}", w=w, h=h, split=split)
            # copy the image alongside so the dataset is self-contained
            dst = out_root / "images" / path.name
            if not dst.exists():
                dst.write_bytes(path.read_bytes())
            ann_path.write_text(ann.model_dump_json(indent=2))
            ok += 1
            logger.info("[%d/%d] %s  (%d regions)", i, len(imgs), stem, len(ann.regions))
        except Exception as e:  # noqa: BLE001
            fail += 1
            logger.warning("[%d/%d] %s FAILED: %s", i, len(imgs), stem, e)
        time.sleep(sleep_s)

    logger.info("Done. ok=%d fail=%d", ok, fail)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--split", default="silver", choices=["silver", "gold", "geom"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--sleep", type=float, default=0.5)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    annotate_dir(a.images, a.out, a.split, a.limit, a.overwrite, a.sleep)


if __name__ == "__main__":
    main()
