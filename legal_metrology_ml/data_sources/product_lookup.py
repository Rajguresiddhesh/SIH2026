"""Resolve a bar code (GTIN) into product declarations using free / public
registries, with an optional licensed GS1 India (DataKart) source.

Design goals
------------
* **Works with zero configuration** — Open Food Facts and its sibling
  databases, UPCItemDB's trial endpoint and Wikidata need no API key.
* **Authoritative source first** — if ``GS1_INDIA_API_KEY`` (and, optionally,
  ``GS1_INDIA_API_URL``) is configured, GS1 India / DataKart is queried first
  and its values win.
* **Never throws** — every network call is defensive; a dead source just
  contributes nothing.
* **Provenance is kept** — ``ProductRecord.field_sources`` records which source
  supplied each field so the compliance report can cite it.
"""

from __future__ import annotations

import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

try:
    import requests
except ImportError:  # keep the module importable without the optional dep
    requests = None  # type: ignore

logger = logging.getLogger(__name__)

_UA = {"User-Agent": "LegalMetrologyComplianceApp/3.0 (compliance-audit)"}
_TIMEOUT = 5           # per-request timeout
_TOTAL_DEADLINE = 12   # overall budget for the whole multi-source lookup

# Source identifiers, in descending order of trust.
SOURCE_GS1_INDIA = "gs1_india_datakart"
SOURCE_OFF = "open_food_facts"
SOURCE_OBF = "open_beauty_facts"
SOURCE_OPF = "open_products_facts"
SOURCE_OPFF = "open_pet_food_facts"
SOURCE_UPCITEMDB = "upcitemdb"
SOURCE_WIKIDATA = "wikidata"

_SOURCE_PRIORITY = [
    SOURCE_GS1_INDIA,
    SOURCE_OFF,
    SOURCE_OBF,
    SOURCE_OPF,
    SOURCE_OPFF,
    SOURCE_UPCITEMDB,
    SOURCE_WIKIDATA,
]

_QTY_RE = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*"
    r"(kg|kgs|g|gm|gms|gram|grams|mg|l|lt|ltr|litre|liter|liters|ml|cl|pcs|pc|piece|pieces|n|nos|units?|u|m|cm|mm)\b",
    re.IGNORECASE,
)
_MRP_RE = re.compile(r"(?:₹|rs\.?|inr|mrp)\s*[:\-]?\s*(\d+(?:\.\d{1,2})?)", re.IGNORECASE)


# ─────────────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ProductRecord:
    """Merged view of a product's Legal-Metrology-relevant declarations,
    assembled from one or more registries."""

    gtin: str

    product_name: Optional[str] = None
    brand: Optional[str] = None

    manufacturer_name: Optional[str] = None
    manufacturer_address: Optional[str] = None
    packer_name: Optional[str] = None
    packer_address: Optional[str] = None
    importer_name: Optional[str] = None
    importer_address: Optional[str] = None
    country_of_origin: Optional[str] = None

    net_quantity_raw: Optional[str] = None
    net_quantity_value: Optional[float] = None
    net_quantity_unit: Optional[str] = None

    mrp_value: Optional[float] = None
    mrp_currency: Optional[str] = None

    manufacture_date: Optional[str] = None
    expiry_date: Optional[str] = None
    best_before: Optional[str] = None

    fssai_license: Optional[str] = None
    veg_non_veg: Optional[str] = None            # "VEG" | "NON_VEG" | None
    consumer_care: Optional[str] = None
    consumer_care_phone: Optional[str] = None
    consumer_care_email: Optional[str] = None

    categories: List[str] = field(default_factory=list)
    ingredients: Optional[str] = None
    image_urls: List[str] = field(default_factory=list)

    # provenance / diagnostics
    sources: List[str] = field(default_factory=list)
    field_sources: Dict[str, str] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def found(self) -> bool:
        """True when at least one registry positively identified the product."""
        return bool(self.sources) and any(
            getattr(self, f) for f in ("product_name", "brand", "net_quantity_raw", "manufacturer_name")
        )

    # -- internal helpers ---------------------------------------------------
    def _set(self, key: str, value: Any, source: str) -> None:
        if value in (None, "", [], {}):
            return
        current = getattr(self, key, None)
        if current in (None, "", [], {}):
            setattr(self, key, value)
            self.field_sources[key] = source

    def to_dict(self) -> dict:
        d = {
            k: getattr(self, k)
            for k in (
                "gtin", "product_name", "brand", "manufacturer_name", "manufacturer_address",
                "packer_name", "packer_address", "importer_name", "importer_address",
                "country_of_origin", "net_quantity_raw", "net_quantity_value", "net_quantity_unit",
                "mrp_value", "mrp_currency", "manufacture_date", "expiry_date", "best_before",
                "fssai_license", "veg_non_veg", "consumer_care", "consumer_care_phone",
                "consumer_care_email", "categories", "ingredients", "image_urls",
            )
        }
        d["found"] = self.found
        d["sources"] = self.sources
        d["field_sources"] = self.field_sources
        return d


# ─────────────────────────────────────────────────────────────────────────────
# Parsing helpers
# ─────────────────────────────────────────────────────────────────────────────
_UNIT_CANON = {
    "kg": "kg", "kgs": "kg",
    "g": "g", "gm": "g", "gms": "g", "gram": "g", "grams": "g",
    "mg": "mg",
    "l": "l", "lt": "l", "ltr": "l", "litre": "l", "liter": "l", "liters": "l",
    "ml": "ml", "cl": "cl",
    "pcs": "U", "pc": "U", "piece": "U", "pieces": "U", "u": "U",
    "n": "N", "nos": "N", "unit": "U", "units": "U",
    "m": "m", "cm": "cm", "mm": "mm",
}


def parse_quantity(text: Optional[str]) -> tuple[Optional[float], Optional[str], Optional[str]]:
    """'500 g' / '1,5 L' / '6 x 30 g' -> (value, canonical_unit, raw_match)."""
    if not text:
        return None, None, None
    m = _QTY_RE.search(str(text))
    if not m:
        return None, None, str(text).strip() or None
    try:
        value = float(m.group(1).replace(",", "."))
    except ValueError:
        return None, None, m.group(0)
    unit = _UNIT_CANON.get(m.group(2).lower(), m.group(2).lower())
    return value, unit, m.group(0)


def parse_mrp(*texts: Optional[str]) -> Optional[float]:
    for t in texts:
        if not t:
            continue
        m = _MRP_RE.search(str(t))
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass
    return None


def _first(*vals: Any) -> Optional[Any]:
    for v in vals:
        if v not in (None, "", [], {}):
            return v
    return None


def _clean(s: Any) -> Optional[str]:
    if s is None:
        return None
    s = str(s).strip().strip(",")
    return s or None


# ─────────────────────────────────────────────────────────────────────────────
# Individual source adapters — each returns a normalised dict or {}
# ─────────────────────────────────────────────────────────────────────────────
def _openfacts(gtin: str, host: str, source: str) -> Dict[str, Any]:
    url = f"https://{host}/api/v2/product/{gtin}.json"
    try:
        resp = requests.get(url, headers=_UA, timeout=_TIMEOUT)
        if resp.status_code != 200:
            return {}
        data = resp.json()
    except Exception as e:  # noqa: BLE001
        logger.debug("%s lookup failed for %s: %s", source, gtin, e)
        return {}

    status = data.get("status")
    if status not in (1, "success", "success_with_warnings"):
        return {}
    p = data.get("product", {}) or {}
    if not p:
        return {}

    qty_val, qty_unit, qty_raw = parse_quantity(p.get("quantity"))

    origin = _clean(_first(
        p.get("origin"),
        ", ".join(t.split(":")[-1].replace("-", " ").title() for t in (p.get("origins_tags") or [])) or None,
        ", ".join(t.split(":")[-1].replace("-", " ").title() for t in (p.get("manufacturing_places_tags") or [])) or None,
    ))

    veg = None
    labels = " ".join(p.get("labels_tags") or []) + " " + " ".join(p.get("ingredients_analysis_tags") or [])
    if "non-vegetarian" in labels or "en:non-vegetarian" in labels:
        veg = "NON_VEG"
    elif "vegetarian" in labels or "vegan" in labels:
        veg = "VEG"

    out = {
        "_source": source,
        "_raw": {k: p.get(k) for k in ("product_name", "brands", "quantity", "countries", "labels_tags")},
        "product_name": _clean(_first(p.get("product_name"), p.get("product_name_en"), p.get("generic_name"))),
        "brand": _clean((p.get("brands") or "").split(",")[0]),
        "manufacturer_name": _clean(_first(
            p.get("manufacturer"),
            (p.get("brands") or "").split(",")[0],
        )),
        "manufacturer_address": _clean(_first(p.get("manufacturing_places"), p.get("emb_codes"))),
        "country_of_origin": origin,
        "net_quantity_raw": qty_raw,
        "net_quantity_value": qty_val,
        "net_quantity_unit": qty_unit,
        "categories": [c.strip() for c in (p.get("categories") or "").split(",") if c.strip()][:6],
        "ingredients": _clean(p.get("ingredients_text")),
        "veg_non_veg": veg,
        "consumer_care": _clean(p.get("customer_service")),
        "image_urls": [u for u in [p.get("image_front_url"), p.get("image_url")] if u],
        "mrp_value": parse_mrp(p.get("stores"), p.get("labels")),
    }
    return out


def _upcitemdb(gtin: str) -> Dict[str, Any]:
    url = f"https://api.upcitemdb.com/prod/trial/lookup?upc={gtin}"
    try:
        resp = requests.get(url, headers=_UA, timeout=_TIMEOUT)
        if resp.status_code != 200:
            return {}
        data = resp.json()
    except Exception as e:  # noqa: BLE001
        logger.debug("upcitemdb lookup failed for %s: %s", gtin, e)
        return {}

    items = data.get("items") or []
    if not items:
        return {}
    it = items[0]
    qty_val, qty_unit, qty_raw = parse_quantity(_first(it.get("size"), it.get("weight"), it.get("dimension")))
    return {
        "_source": SOURCE_UPCITEMDB,
        "_raw": {k: it.get(k) for k in ("title", "brand", "category", "size", "weight")},
        "product_name": _clean(it.get("title")),
        "brand": _clean(it.get("brand")),
        "manufacturer_name": _clean(_first(it.get("manufacturer"), it.get("brand"))),
        "net_quantity_raw": qty_raw,
        "net_quantity_value": qty_val,
        "net_quantity_unit": qty_unit,
        "categories": [c.strip() for c in (it.get("category") or "").split(">") if c.strip()][:6],
        "image_urls": it.get("images") or [],
    }


def _wikidata(gtin: str) -> Dict[str, Any]:
    # P3962 = Global Trade Item Number. Match the bare number and common
    # zero-padded / EAN-13 forms.
    forms = {gtin, gtin.lstrip("0"), gtin.zfill(13), gtin.zfill(14)}
    values = " ".join(f'"{f}"' for f in forms if f)
    query = f"""
    SELECT ?itemLabel ?brandLabel ?manufacturerLabel ?countryLabel WHERE {{
      VALUES ?gtin {{ {values} }}
      ?item wdt:P3962 ?gtin .
      OPTIONAL {{ ?item wdt:P1716 ?brand. }}
      OPTIONAL {{ ?item wdt:P176 ?manufacturer. }}
      OPTIONAL {{ ?item wdt:P495 ?country. }}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }} LIMIT 1
    """
    try:
        resp = requests.get(
            "https://query.wikidata.org/sparql",
            params={"query": query, "format": "json"},
            headers=_UA,
            timeout=_TIMEOUT,
        )
        if resp.status_code != 200:
            return {}
        rows = resp.json().get("results", {}).get("bindings", [])
    except Exception as e:  # noqa: BLE001
        logger.debug("wikidata lookup failed for %s: %s", gtin, e)
        return {}

    if not rows:
        return {}
    r = rows[0]

    def g(key: str) -> Optional[str]:
        return _clean(r.get(key, {}).get("value"))

    return {
        "_source": SOURCE_WIKIDATA,
        "_raw": {k: g(k) for k in ("itemLabel", "brandLabel", "manufacturerLabel", "countryLabel")},
        "product_name": g("itemLabel"),
        "brand": g("brandLabel"),
        "manufacturer_name": g("manufacturerLabel"),
        "country_of_origin": g("countryLabel"),
    }


def _gs1_india(gtin: str) -> Dict[str, Any]:
    """GS1 India / DataKart ('Verified by GS1' / Smart Consumer backend).

    Requires a licensed subscription.  Configure via environment:

        GS1_INDIA_API_KEY   – subscription / bearer key (required to enable)
        GS1_INDIA_API_URL   – override endpoint; ``{gtin}`` is substituted
                              (default: DataKart public GTIN verification API)
    """
    api_key = os.environ.get("GS1_INDIA_API_KEY", "").strip()
    if not api_key:
        return {}

    url_tmpl = os.environ.get(
        "GS1_INDIA_API_URL",
        "https://datakartapigateway.gs1india.org/api/v1/product/gtin/{gtin}",
    )
    url = url_tmpl.replace("{gtin}", gtin)
    headers = dict(_UA)
    headers["Authorization"] = f"Bearer {api_key}"
    headers["APIKey"] = api_key  # DataKart historically accepts an APIKey header

    try:
        resp = requests.get(url, headers=headers, timeout=_TIMEOUT)
        if resp.status_code != 200:
            logger.info("GS1 India returned HTTP %s for %s", resp.status_code, gtin)
            return {}
        data = resp.json()
    except Exception as e:  # noqa: BLE001
        logger.warning("GS1 India lookup failed for %s: %s", gtin, e)
        return {}

    # DataKart nests the product under several possible keys across versions.
    p = _first(
        data.get("product"),
        (data.get("products") or [None])[0],
        data.get("data"),
        data,
    ) or {}
    if not isinstance(p, dict):
        return {}

    qty_val, qty_unit, qty_raw = parse_quantity(
        _first(p.get("netContent"), p.get("net_content"), p.get("netQuantity"), p.get("quantity"))
    )
    company = p.get("companyName") or p.get("company_name") or p.get("brandOwner")
    addr = _first(p.get("companyAddress"), p.get("company_address"), p.get("manufacturerAddress"))

    return {
        "_source": SOURCE_GS1_INDIA,
        "_raw": data if len(str(data)) < 4000 else {"note": "payload truncated"},
        "product_name": _clean(_first(p.get("productName"), p.get("product_name"), p.get("productDescription"))),
        "brand": _clean(_first(p.get("brandName"), p.get("brand_name"), p.get("brand"))),
        "manufacturer_name": _clean(_first(p.get("manufacturedBy"), p.get("manufacturer"), company)),
        "manufacturer_address": _clean(_first(p.get("manufacturedByAddress"), addr)),
        "packer_name": _clean(_first(p.get("packedBy"), p.get("marketedBy"))),
        "packer_address": _clean(p.get("packedByAddress")),
        "importer_name": _clean(p.get("importedBy")),
        "importer_address": _clean(p.get("importedByAddress")),
        "country_of_origin": _clean(_first(p.get("countryOfOrigin"), p.get("country_of_origin"), "India")),
        "net_quantity_raw": qty_raw,
        "net_quantity_value": qty_val,
        "net_quantity_unit": qty_unit,
        "mrp_value": parse_mrp(str(p.get("mrp")), str(p.get("MRP"))) or (
            float(p["mrp"]) if str(p.get("mrp", "")).replace(".", "", 1).isdigit() else None
        ),
        "mrp_currency": "INR",
        "manufacture_date": _clean(_first(p.get("manufacturingDate"), p.get("packagingDate"))),
        "best_before": _clean(p.get("bestBefore")),
        "fssai_license": _clean(_first(p.get("fssaiLicenseNo"), p.get("fssai"))),
        "veg_non_veg": (
            "VEG" if str(p.get("vegNonVeg", "")).lower().startswith("veg")
            else "NON_VEG" if "non" in str(p.get("vegNonVeg", "")).lower()
            else None
        ),
        "consumer_care": _clean(_first(p.get("customerCareDetails"), p.get("consumerCare"))),
        "consumer_care_phone": _clean(_first(p.get("customerCareNumber"), p.get("consumerCarePhone"))),
        "consumer_care_email": _clean(_first(p.get("customerCareEmail"), p.get("consumerCareEmail"))),
        "image_urls": [u for u in (p.get("frontImage"), p.get("imageUrl"), p.get("image")) if u],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Aggregator
# ─────────────────────────────────────────────────────────────────────────────
_MERGEABLE_FIELDS = [
    "product_name", "brand", "manufacturer_name", "manufacturer_address",
    "packer_name", "packer_address", "importer_name", "importer_address",
    "country_of_origin", "net_quantity_raw", "net_quantity_value", "net_quantity_unit",
    "mrp_value", "mrp_currency", "manufacture_date", "expiry_date", "best_before",
    "fssai_license", "veg_non_veg", "consumer_care", "consumer_care_phone",
    "consumer_care_email", "ingredients",
]


def lookup_product(gtin: str, *, enable_network: bool = True) -> ProductRecord:
    """Query every configured registry for *gtin* and return a merged
    :class:`ProductRecord`.  Higher-trust sources are applied first so their
    values win on conflict."""
    digits = "".join(c for c in str(gtin) if c.isdigit())
    record = ProductRecord(gtin=digits or str(gtin))

    if not enable_network or not digits or requests is None:
        if requests is None:
            logger.warning("`requests` is not installed — product registry lookup skipped.")
        return record

    adapters: List[Callable[[], Dict[str, Any]]] = [
        lambda: _gs1_india(digits),
        lambda: _openfacts(digits, "world.openfoodfacts.org", SOURCE_OFF),
        lambda: _openfacts(digits, "world.openbeautyfacts.org", SOURCE_OBF),
        lambda: _openfacts(digits, "world.openproductsfacts.org", SOURCE_OPF),
        lambda: _openfacts(digits, "world.openpetfoodfacts.org", SOURCE_OPFF),
        lambda: _upcitemdb(digits),
        lambda: _wikidata(digits),
    ]

    def _safe_run(fn: Callable[[], Dict[str, Any]]) -> Dict[str, Any]:
        try:
            return fn() or {}
        except Exception as e:  # noqa: BLE001 — a source must never break the audit
            logger.debug("product source raised: %s", e)
            return {}

    payloads: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=len(adapters)) as pool:
        futures = {pool.submit(_safe_run, fn): i for i, fn in enumerate(adapters)}
        try:
            for fut in as_completed(futures, timeout=_TOTAL_DEADLINE):
                res = fut.result()
                if res:
                    payloads.append(res)
        except Exception as e:  # noqa: BLE001 — overall deadline hit
            logger.info("Product lookup deadline reached for %s (%s); using partial results.", digits, e)

    # Apply in priority order.
    payloads.sort(key=lambda d: _SOURCE_PRIORITY.index(d.get("_source", SOURCE_WIKIDATA))
                  if d.get("_source") in _SOURCE_PRIORITY else 99)

    for payload in payloads:
        source = payload.get("_source", "unknown")
        record.sources.append(source)
        record.raw[source] = payload.get("_raw")
        for f in _MERGEABLE_FIELDS:
            record._set(f, payload.get(f), source)
        for cat in payload.get("categories", []) or []:
            if cat not in record.categories:
                record.categories.append(cat)
        for img in payload.get("image_urls", []) or []:
            if img not in record.image_urls:
                record.image_urls.append(img)

    # Derive net quantity components if only the raw string came through.
    if record.net_quantity_raw and record.net_quantity_value is None:
        v, u, _ = parse_quantity(record.net_quantity_raw)
        if v is not None:
            record.net_quantity_value = v
            record.field_sources.setdefault("net_quantity_value", record.field_sources.get("net_quantity_raw", "derived"))
        if u:
            record.net_quantity_unit = u

    return record
