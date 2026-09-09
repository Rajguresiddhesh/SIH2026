"""Synthetic PCR-Label generator.

Renders procedurally-composed package labels with *exactly* known declaration
boxes, transcribed values and glyph geometry (fixed mm/px). Used for:
  - pre-training the visual extractor before the human gold set exists,
  - a runnable end-to-end sanity harness (eval + conformal calibration),
  - controlled robustness / violation studies.

Not a substitute for the real `gold` / `geom` splits used in the paper's
evaluation — synthetic labels are marked `split="synth"` and excluded from the
reported test metrics.

    python -m ml.data.synth --out data/pcr_label --n 400 --geom-frac 0.35 --seed 0
"""
from __future__ import annotations

import argparse
import random
import string
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from ml.data.schema import BBox, GeometryGT, GlyphMetric, ImageAnnotation, Region
from ml.data.taxonomy import DeclType, min_letter_height_mm


def _find_fonts() -> Path:
    import matplotlib

    return Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf"


_FONT_DIR = _find_fonts()
_FONTS = [str(_FONT_DIR / f) for f in ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf")]

MM_PER_PX = 0.15  # fixed synthetic scale


def _gtin13(rng: random.Random, indian: bool = True) -> str:
    body = ("890" if indian else f"{rng.randint(0,899):03d}") + "".join(
        rng.choice(string.digits) for _ in range(9)
    )
    d = [int(c) for c in body]
    chk = (10 - (sum(d[0::2]) + 3 * sum(d[1::2])) % 10) % 10
    return body + str(chk)


def _draw_barcode(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int, rng: random.Random) -> None:
    n = 40
    bw = w / n
    for i in range(n):
        if rng.random() < 0.5:
            draw.rectangle([x + i * bw, y, x + i * bw + bw * 0.8, y + h], fill=(0, 0, 0))


def _cap_height_px(font: ImageFont.FreeTypeFont) -> float:
    box = font.getbbox("MHE1")
    return float(box[3] - box[1])


def _clamp01(v: float) -> float:
    return min(1.0, max(0.0, v))


def _fit(text: str, font: ImageFont.FreeTypeFont, max_w: int) -> str:
    if font.getbbox(text)[2] <= max_w:
        return text
    while text and font.getbbox(text + "...")[2] > max_w:
        text = text[:-1]
    return (text + "...") if text else text


def _place_text(
    draw: ImageDraw.ImageDraw, img_w: int, img_h: int, x: int, y: int,
    text: str, font: ImageFont.FreeTypeFont, max_w: int | None = None,
) -> BBox:
    if max_w:
        text = _fit(text, font, max_w)
    draw.text((x, y), text, fill=(10, 10, 10), font=font)
    tb = draw.textbbox((x, y), text, font=font)
    return BBox(
        x0=_clamp01(tb[0] / img_w), y0=_clamp01(tb[1] / img_h),
        x1=_clamp01(tb[2] / img_w), y1=_clamp01(tb[3] / img_h),
    )


def _rand_date(rng: random.Random) -> str:
    m = rng.choice(["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"])
    return f"{m} {rng.randint(24, 27)}"


def generate_one(rng: random.Random, want_geom: bool, idx: int) -> tuple[Image.Image, ImageAnnotation]:
    W = rng.randint(560, 820)
    H = rng.randint(760, 1180)
    bg = tuple(rng.randint(200, 255) for _ in range(3))
    img = Image.new("RGB", (W, H), bg)
    draw = ImageDraw.Draw(img)

    # PDP rectangle
    m = int(0.05 * W)
    pdp = (m, m, W - m, H - m)
    draw.rectangle(pdp, outline=(150, 150, 150), width=2)
    pdp_box = BBox(x0=pdp[0] / W, y0=pdp[1] / H, x1=pdp[2] / W, y1=pdp[3] / H)
    pdp_area_cm2 = ((pdp[2] - pdp[0]) * MM_PER_PX) * ((pdp[3] - pdp[1]) * MM_PER_PX) / 100.0
    min_h_mm = min_letter_height_mm(pdp_area_cm2)

    regions: list[Region] = []
    glyphs: list[GlyphMetric] = []

    # violation knobs
    tiny_font = rng.random() < 0.35            # -> Rule 9 FAIL
    drop_tax_clause = rng.random() < 0.4       # -> R06_MRP_TAX FAIL/INCON
    drop_consumer = rng.random() < 0.3
    imported = rng.random() < 0.25

    # normal font size = a bit above the minimum; tiny = below
    ok_px = (min_h_mm / MM_PER_PX) * rng.uniform(1.15, 1.8)
    bad_px = (min_h_mm / MM_PER_PX) * rng.uniform(0.4, 0.85)
    body_font_px = int(bad_px if tiny_font else ok_px)
    body_font_px = max(10, min(body_font_px, 46))
    font = ImageFont.truetype(rng.choice(_FONTS), body_font_px)
    big = ImageFont.truetype(_FONTS[1], int(body_font_px * 1.6))

    gtin = _gtin13(rng, indian=not imported)
    y = pdp[1] + 20
    x = pdp[0] + 20
    cw = pdp[2] - 20 - x  # content width in px

    name = rng.choice(["Toor Dal", "Multigrain Atta", "Almond Drink", "Choco Cookies",
                       "Basmati Rice", "Green Tea", "Hair Oil", "Dish Wash Gel"])
    regions.append(Region(cls=DeclType.GENERIC_NAME.value,
                          bbox=_place_text(draw, W, H, x, y, name, big, cw), value=name, source="human"))
    y += int(body_font_px * 2.2)

    qty = rng.choice([("Net Wt. 500 g", 500), ("Net Qty 200 g", 200), ("Net Vol. 1 L", 1000),
                      ("Net Wt. 100 g", 100), ("30 N", 30)])
    b = _place_text(draw, W, H, x, y, qty[0], font, cw)
    regions.append(Region(cls=DeclType.NET_QUANTITY.value, bbox=b, value=qty[0], source="human"))
    glyphs.append(GlyphMetric(region_cls=DeclType.NET_QUANTITY.value,
                              cap_line_y=b.y0, base_line_y=b.y1,
                              cap_height_mm=round(_cap_height_px(font) * MM_PER_PX, 3)))
    y += int(body_font_px * 1.8)

    mrp_txt = f"MRP Rs. {rng.randint(20, 900)}.00" + ("" if drop_tax_clause else " (incl. of all taxes)")
    b = _place_text(draw, W, H, x, y, mrp_txt, font, cw)
    regions.append(Region(cls=DeclType.MRP.value, bbox=b, value=mrp_txt, source="human"))
    glyphs.append(GlyphMetric(region_cls=DeclType.MRP.value, cap_line_y=b.y0, base_line_y=b.y1,
                              cap_height_mm=round(_cap_height_px(font) * MM_PER_PX, 3)))
    if not drop_tax_clause:
        regions.append(Region(cls=DeclType.MRP_TAX_CLAUSE.value, bbox=b, value="incl. of all taxes",
                              source="human"))
    y += int(body_font_px * 1.8)

    d = _rand_date(rng)
    regions.append(Region(cls=DeclType.MFG_DATE.value,
                          bbox=_place_text(draw, W, H, x, y, f"Mfd: {d}", font, cw),
                          value=f"Mfd: {d}", source="human"))
    y += int(body_font_px * 1.7)

    mfr = rng.choice(["Bharat Foods Pvt Ltd", "Ganga Mills Ltd", "Sunrise Agro Pvt Ltd"])
    regions.append(Region(cls=DeclType.MANUFACTURER_NAME.value,
                          bbox=_place_text(draw, W, H, x, y, mfr, font, cw), value=mfr, source="human"))
    y += int(body_font_px * 1.6)
    addr = f"Plot {rng.randint(1,99)}, MIDC, Pune {rng.randint(410000,411999)}, Maharashtra, India"
    regions.append(Region(cls=DeclType.MANUFACTURER_ADDRESS.value,
                          bbox=_place_text(draw, W, H, x, y, addr, font, cw), value=addr, source="human"))
    y += int(body_font_px * 1.7)

    if imported:
        regions.append(Region(cls=DeclType.PACKER_IMPORTER.value,
                              bbox=_place_text(draw, W, H, x, y, "Imported by Global Traders, Mumbai", font, cw),
                              value="Imported by Global Traders, Mumbai", source="human"))
        y += int(body_font_px * 1.6)
        regions.append(Region(cls=DeclType.COUNTRY_OF_ORIGIN.value,
                              bbox=_place_text(draw, W, H, x, y, "Country of Origin: Germany", font, cw),
                              value="Country of Origin: Germany", source="human"))
        y += int(body_font_px * 1.7)

    if not drop_consumer:
        cc = f"Consumer Care: care@brand.in, 1800-{rng.randint(100,999)}-{rng.randint(1000,9999)}"
        regions.append(Region(cls=DeclType.CONSUMER_CARE.value,
                              bbox=_place_text(draw, W, H, x, y, cc, font, cw), value=cc, source="human"))
        y += int(body_font_px * 1.7)

    fssai = "FSSAI " + "".join(rng.choice(string.digits) for _ in range(14))
    regions.append(Region(cls=DeclType.FSSAI.value,
                          bbox=_place_text(draw, W, H, x, y, fssai, font, cw), value=fssai, source="human"))

    # veg/nonveg mark
    mk = pdp[2] - 60
    mky = pdp[1] + 20
    if rng.random() < 0.5:
        draw.rectangle([mk, mky, mk + 26, mky + 26], outline=(0, 120, 0), width=2)
        draw.ellipse([mk + 6, mky + 6, mk + 20, mky + 20], fill=(0, 140, 0))
        regions.append(Region(cls=DeclType.VEG_MARK.value,
                              bbox=BBox(x0=mk / W, y0=mky / H, x1=(mk + 26) / W, y1=(mky + 26) / H),
                              source="human"))
    else:
        draw.rectangle([mk, mky, mk + 26, mky + 26], outline=(140, 60, 0), width=2)
        draw.polygon([(mk + 13, mky + 5), (mk + 5, mky + 21), (mk + 21, mky + 21)], fill=(150, 60, 0))
        regions.append(Region(cls=DeclType.NONVEG_MARK.value,
                              bbox=BBox(x0=mk / W, y0=mky / H, x1=(mk + 26) / W, y1=(mky + 26) / H),
                              source="human"))

    # barcode near bottom
    bx, by, bw, bh = pdp[0] + 30, pdp[3] - 90, 200, 60
    _draw_barcode(draw, bx, by, bw, bh, rng)
    draw.text((bx, by + bh + 2), gtin, fill=(0, 0, 0),
              font=ImageFont.truetype(_FONTS[0], 14))
    regions.append(Region(cls=DeclType.BARCODE.value,
                          bbox=BBox(x0=bx / W, y0=by / H, x1=(bx + bw) / W, y1=(by + bh) / H),
                          value=gtin, source="human"))

    regions.append(Region(cls=DeclType.PRINCIPAL_DISPLAY_PANEL.value, bbox=pdp_box, source="human"))

    geometry = None
    if want_geom:
        # a "ruler": a horizontal 100 mm span drawn faintly at the bottom margin
        r_y = H - 14
        r_len_px = int(100 / MM_PER_PX)
        r_x0 = 10
        draw.line([(r_x0, r_y), (r_x0 + r_len_px, r_y)], fill=(60, 60, 60), width=2)
        geometry = GeometryGT(
            ruler_p0=(r_x0 / W, r_y / H), ruler_p1=((r_x0 + r_len_px) / W, r_y / H),
            ruler_span_mm=100.0, mm_per_px=MM_PER_PX,
            pdp_area_cm2=round(pdp_area_cm2, 1), glyphs=glyphs,
        )

    ann = ImageAnnotation(
        image=f"images/synth_{idx:05d}.jpg", width=W, height=H,
        split="geom" if want_geom else "synth", view="front",
        package_type=rng.choice(["pouch", "carton", "box", "bottle"]),
        is_imported=imported, regions=regions, geometry=geometry,
        annotator_notes=f"synthetic; min_h_mm={min_h_mm}; tiny_font={tiny_font}; "
                        f"drop_tax={drop_tax_clause}; drop_consumer={drop_consumer}",
    )
    return img, ann


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--geom-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    (a.out / "images").mkdir(parents=True, exist_ok=True)
    (a.out / "annotations").mkdir(parents=True, exist_ok=True)

    for i in range(a.n):
        want_geom = rng.random() < a.geom_frac
        img, ann = generate_one(rng, want_geom, i)
        img.save(a.out / "images" / f"synth_{i:05d}.jpg", quality=92)
        (a.out / "annotations" / f"synth_{i:05d}.json").write_text(ann.model_dump_json(indent=2))
        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{a.n}")

    from ml.data.dataset import PcrLabelDataset

    ds = PcrLabelDataset(a.out)
    ds.save_index()
    print(f"done -> {a.out}  splits={ {k: len(v) for k, v in ds.index.splits.items()} }")


if __name__ == "__main__":
    main()
