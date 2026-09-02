from __future__ import annotations

from typing import Optional, List, Dict
from enum import Enum
from pydantic import BaseModel, Field

class QuantityCategory(str, Enum):
    WEIGHT = 'weight'
    VOLUME = 'volume'
    LENGTH = 'length'
    AREA = 'area'
    NUMBER = 'number'

class PackageType(str, Enum):
    RETAIL = 'retail'
    WHOLESALE = 'wholesale'
    EXPORT = 'export'

class PackageData(BaseModel):
    """Normalized data extracted from a packaged commodity label.
    Maps to declarations required under Legal Metrology (Packaged Commodities) Rules, 2011."""
    
    # Identity (Rule 6)
    commodity_name: Optional[str] = None
    commodity_name_confidence: float = 0.0
    
    # Manufacturer/Packer/Importer (Rule 10)
    manufacturer_name: Optional[str] = None
    manufacturer_address: Optional[str] = None
    packer_name: Optional[str] = None
    packer_address: Optional[str] = None
    importer_name: Optional[str] = None
    importer_address: Optional[str] = None
    country_of_origin: Optional[str] = None
    
    # Net Quantity (Rules 11-13)
    net_quantity_value: Optional[float] = None
    net_quantity_unit: Optional[str] = None
    net_quantity_category: Optional[QuantityCategory] = None
    
    # MRP (Rule 18)
    mrp_value: Optional[float] = None
    mrp_currency: Optional[str] = None
    mrp_includes_tax: Optional[bool] = None
    mrp_altered: Optional[bool] = None
    
    # Date (Rule 6)
    manufacture_date: Optional[str] = None
    expiry_date: Optional[str] = None
    best_before: Optional[str] = None
    
    # Consumer Care (Rule 6)
    consumer_care_name: Optional[str] = None
    consumer_care_phone: Optional[str] = None
    consumer_care_email: Optional[str] = None
    consumer_care_address: Optional[str] = None
    
    # Symbols & Logos
    fssai_license_number: Optional[str] = None
    has_fssai_logo: bool = False
    has_veg_nonveg_symbol: bool = False
    has_isi_mark: bool = False
    has_recycling_symbol: bool = False
    has_barcode: bool = False
    barcode_value: Optional[str] = None
    barcode_type: Optional[str] = None
    barcode_valid: Optional[bool] = None
    barcode_country: Optional[str] = None
    barcode_gtin_format: Optional[str] = None       # GTIN-8 / UPC-A / EAN-13 / GTIN-14
    barcode_checksum_valid: Optional[bool] = None   # GS1 mod-10 check digit
    barcode_is_gs1_india: Optional[bool] = None     # prefix 890 -> GS1 India licence
    barcode_is_restricted: Optional[bool] = None    # retailer-internal / coupon / bookland
    barcode_registered_owner: Optional[str] = None  # brand owner / company from registry
    
    # Font/Presentation Metrics (Rules 7-9)
    mrp_font_height_mm: Optional[float] = None
    net_qty_font_height_mm: Optional[float] = None
    min_font_height_mm: Optional[float] = None
    mrp_contrast_ratio: Optional[float] = None
    net_qty_contrast_ratio: Optional[float] = None
    
    # Spatial Metrics (Rule 8)
    net_qty_clear_space_above_mm: Optional[float] = None
    net_qty_clear_space_below_mm: Optional[float] = None
    net_qty_clear_space_left_mm: Optional[float] = None
    net_qty_clear_space_right_mm: Optional[float] = None
    
    # Panel Metrics
    principal_display_panel_area_mm2: Optional[float] = None
    declarations_on_principal_panel: bool = False
    
    # Language (Rule 9)
    has_hindi_text: bool = False
    has_english_text: bool = False
    
    # Package Meta
    is_imported: bool = False
    is_multicomponent: bool = False
    package_type: PackageType = PackageType.RETAIL
    
    # OCR Quality
    average_ocr_confidence: float = 0.0
    total_text_blocks: int = 0

    # Provenance — how this record was assembled
    analysis_source: str = "image"            # image | barcode_registry | llm_vision | llm_barcode
    product_data_sources: List[str] = Field(default_factory=list)
    data_provenance: Dict[str, str] = Field(default_factory=dict)  # field -> source
    product_identified: bool = False          # a registry positively resolved the GTIN

class RuleResult(BaseModel):
    """Result of a single compliance rule check."""
    rule_id: str
    rule_name: str
    status: str  # PASS, FAIL, WARNING, NOT_APPLICABLE, INCONCLUSIVE
    detail: str
    severity: str  # CRITICAL, MAJOR, MINOR
    weight: float
    evidence: Optional[str] = None
    legal_reference: Optional[str] = None

class ComplianceDiff(BaseModel):
    """Results from running all rulebook checks."""
    total_rules: int
    passed: List[RuleResult]
    failed: List[RuleResult]
    warnings: List[RuleResult]
    not_applicable: List[RuleResult]
    inconclusive: List[RuleResult]

class CompliancePrediction(BaseModel):
    """ML model prediction with explanation."""
    compliance_probability: float
    predicted_compliant: bool
    feature_contributions: Dict[str, float]  # feature_name -> contribution_score
    top_risk_factors: List[str]

class ComplianceScore(BaseModel):
    """Final aggregated compliance assessment."""
    final_score: float
    star_rating: int
    star_label: str
    ebm_score: float
    rule_score: float
    total_applicable_rules: int
    passed_rules: int
    failed_rules: int
    critical_failures: int
    major_failures: int
    minor_failures: int

class ComplianceReport(BaseModel):
    """Complete compliance report."""
    scan_id: str
    scan_timestamp: str
    image_path: str
    package_data: PackageData
    ebm_prediction: CompliancePrediction
    rulebook_diff: ComplianceDiff
    compliance_score: ComplianceScore
    recommendations: List[str]
