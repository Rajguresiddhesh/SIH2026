from __future__ import annotations

"""External product-data sources used to resolve a bar code (GTIN) into the
declarations that the Legal Metrology (Packaged Commodities) Rules, 2011
require on the package."""

from .product_lookup import ProductRecord, lookup_product

__all__ = ["ProductRecord", "lookup_product"]
