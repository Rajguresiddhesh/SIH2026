"""PCR-Label annotation schema.

One JSON file per image: `<dataset>/annotations/<image_stem>.json`. Validated
on load; the Gemini annotator and the human-review tool both emit this shape.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, field_validator

from .taxonomy import CLASSES


class BBox(BaseModel):
    """Axis-aligned box in **normalised** image coordinates (0–1), x0<x1, y0<y1."""

    x0: float = Field(ge=0.0, le=1.0)
    y0: float = Field(ge=0.0, le=1.0)
    x1: float = Field(ge=0.0, le=1.0)
    y1: float = Field(ge=0.0, le=1.0)

    @field_validator("x1")
    @classmethod
    def _x_order(cls, v: float, info) -> float:
        if v < info.data.get("x0", 0.0):
            raise ValueError("x1 < x0")
        return v

    @field_validator("y1")
    @classmethod
    def _y_order(cls, v: float, info) -> float:
        if v < info.data.get("y0", 0.0):
            raise ValueError("y1 < y0")
        return v

    def to_abs(self, w: int, h: int) -> tuple[int, int, int, int]:
        return (round(self.x0 * w), round(self.y0 * h), round(self.x1 * w), round(self.y1 * h))

    @property
    def area(self) -> float:
        return max(0.0, self.x1 - self.x0) * max(0.0, self.y1 - self.y0)


class Region(BaseModel):
    """One detected declaration region."""

    cls: str
    bbox: BBox
    polygon: Optional[list[tuple[float, float]]] = None   # normalised, for PDP / skewed text
    value: Optional[str] = None                           # transcribed text (TEXTUAL classes)
    script: Optional[str] = None                          # "latin" | "devanagari" | "mixed"
    source: str = "human"                                 # "human" | "gemini" | "model"
    confidence: float = 1.0

    @field_validator("cls")
    @classmethod
    def _known_class(cls, v: str) -> str:
        if v not in CLASSES:
            raise ValueError(f"unknown class {v!r}")
        return v


class GlyphMetric(BaseModel):
    """Ground-truth glyph geometry (the `geom` split only)."""

    region_cls: str
    cap_line_y: float           # normalised y of the cap line
    base_line_y: float
    mean_line_y: Optional[float] = None
    cap_height_mm: Optional[float] = None
    x_height_mm: Optional[float] = None
    stroke_width_mm: Optional[float] = None


class GeometryGT(BaseModel):
    """Scale ground truth from a ruler in the frame (the `geom` split)."""

    ruler_p0: tuple[float, float]      # normalised endpoints of a known span
    ruler_p1: tuple[float, float]
    ruler_span_mm: float
    mm_per_px: float                  # derived, at the ruler's image location
    pdp_area_cm2: Optional[float] = None
    glyphs: list[GlyphMetric] = Field(default_factory=list)


class ImageAnnotation(BaseModel):
    image: str                        # relative path under <dataset>/images/
    width: int
    height: int
    split: str = "silver"             # silver | gold | geom
    view: str = "front"               # front | back | side
    package_type: str = "unknown"     # pouch | bottle | can | carton | box | blister | unknown
    is_imported: bool = False
    regions: list[Region] = Field(default_factory=list)
    geometry: Optional[GeometryGT] = None
    annotator_notes: Optional[str] = None
    reviewed_by: Optional[str] = None

    def region_map(self) -> dict[str, list[Region]]:
        out: dict[str, list[Region]] = {}
        for r in self.regions:
            out.setdefault(r.cls, []).append(r)
        return out


class DatasetIndex(BaseModel):
    """`<dataset>/index.json` — the manifest the loaders read."""

    name: str = "PCR-Label"
    version: str = "0.1.0"
    images: list[str] = Field(default_factory=list)     # annotation json stems
    splits: dict[str, list[str]] = Field(default_factory=dict)
    label_map: dict[str, int] = Field(default_factory=dict)
