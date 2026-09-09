# PCR-Label annotation guide

Read before correcting a `gold` or `geom` image. One JSON per image at
`data/pcr_label/annotations/<stem>.json`, shape = `ml/data/schema.py::ImageAnnotation`.

## Coordinate system

Normalised `[0,1]`, origin **top-left**, `x0<x1`, `y0<y1`. A box is the tight
axis-aligned rectangle around the *ink* (not the visual "field label"). For
rotated / curved text use `polygon` (list of normalised `[x,y]`) **and** still
give the enclosing `bbox`.

## The 17 classes

| class | what to box | notes |
|---|---|---|
| `principal_display_panel` | the single largest face that carries brand + net quantity | exactly one per image; the panel outline, not the whole photo |
| `generic_name` | the common/generic name ("Toor Dal", "Multivitamin Tablets") | not the brand mark alone |
| `net_quantity` | the whole "Net Wt. 500 g ⊕" block | include the "Net Wt./Net Qty." words |
| `net_quantity_unit` | just the unit token inside it (`g`,`kg`,`ml`,`N`) | for the Rule 13 unit check |
| `mrp` | the price block ("MRP ₹45.00", "M.R.P. Rs 45/-") | include the "MRP/M.R.P." words |
| `mrp_tax_clause` | "inclusive of all taxes" / "incl. of all taxes" | may be physically separate from `mrp`; omit if absent |
| `mfg_date` | mfg / packing / import date block | "Mfd:", "Pkd:", "Mfg. Date" |
| `expiry_date` | "Use by" / "Best before" / "Expiry" block | duration form ("Best before 9 months") is fine |
| `manufacturer_name` | the name line of the maker/packer/importer | |
| `manufacturer_address` | the address block (may span lines) | box the whole block |
| `packer_importer` | a distinct "Packed by" / "Marketed by" / "Imported by" block | only if separate from `manufacturer_*` |
| `consumer_care` | customer-care block (name/phone/email/address) | |
| `country_of_origin` | "Country of Origin: X" | |
| `fssai` | the FSSAI logo **and** licence number together | one box covering both if adjacent, else the number |
| `veg_mark` | green filled dot inside a green square outline | |
| `nonveg_mark` | brown/red filled triangle inside a square outline | |
| `barcode` | the GS1 bar-code symbol (bars only, not the digit row) | this is also the metric-scale reference |

**Do not** create a box for a class that is not visibly present. Multiple
instances of a class are allowed (e.g. MRP on front and back → annotate each
view's file separately).

## `value` (textual classes)

Verbatim transcription. Keep the original script; **transliterate nothing**.
Keep punctuation and currency symbols. For multi-line blocks join lines with a
single space. Set `script` = `latin` / `devanagari` / `mixed`.

## Image-level fields

- `view`: `front` / `back` / `side`
- `package_type`: `pouch` / `bottle` / `can` / `carton` / `box` / `blister` / `unknown`
- `is_imported`: true if any "Imported by" / foreign country of origin appears
- set `reviewed_by` to your initials, and `split` to `gold`

## `geom` split extras

1. Place a mm ruler flat against the package in the same plane as the panel.
2. `geometry.ruler_p0` / `ruler_p1`: normalised endpoints of a **known** span on
   the ruler; `ruler_span_mm`: that span in mm; `mm_per_px`: computed.
3. For ≥5 glyphs across the height-checked classes, add a `GlyphMetric` with the
   normalised `cap_line_y` / `base_line_y` (and `mean_line_y` for lowercase),
   plus the measured `cap_height_mm`.
4. `pdp_area_cm2`: measure the PDP with the ruler.

## QA

`python -m ml.data.dataset` (import) validates every JSON on load; a schema
error names the offending file. Run `python -m ml.data.split --root ... --auto ...`
after a batch to rebuild `index.json`.
