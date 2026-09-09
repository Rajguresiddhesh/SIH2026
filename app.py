"""
Flask web server for the Legal Metrology Compliance Checker.

Provides two image-input modes:
  1. File upload (drag-and-drop or browse)
  2. Live camera capture (via browser getUserMedia API)

Run with:
    python app.py
Then open http://localhost:5000 in your browser.
"""
from __future__ import annotations

import base64
import io
import json
import logging
import os
import tempfile
import traceback
import uuid
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file

# ---------------------------------------------------------------------------
# Pipeline import
# ---------------------------------------------------------------------------
from legal_metrology_ml.main import run_pipeline
from legal_metrology_ml.layer1_feature_extraction.ocr_engine import OCREngine

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20 MB

UPLOAD_DIR = Path(tempfile.gettempdir()) / "lmc_uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

REPORT_DIR = Path(tempfile.gettempdir()) / "lmc_reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Shared singleton — EasyOCR loads ~200 MB of models; build once at startup
# so every request reuses the same reader instead of reloading it each time.
# ---------------------------------------------------------------------------
logger.info("Loading OCREngine at startup (this takes ~30s on first run)…")
_ocr_engine = OCREngine()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Accepts either multipart/form-data or JSON body.

    Multipart fields:
        front              — required image file
        back               — optional back/side label image file
        ruler              — optional ruler calibration image file
        package_height_mm  — optional float

    JSON body fields:
        front_b64          — required base64 image
        back_b64           — optional base64 image
        ruler_b64          — optional base64 image
        package_height_mm  — optional float
    """
    front_path: Path | None = None
    back_path:  Path | None = None
    ruler_path: Path | None = None
    barcode_path: Path | None = None
    barcode_number: str | None = None
    api_key: str | None = None

    try:
        package_height_mm = None
        api_key = request.headers.get("X-Gemini-Api-Key")
        # engine: auto | ml | llm | local  (query string or form/JSON field)
        engine = (request.args.get("engine")
                  or (request.form.get("engine") if request.form else None)
                  or "auto")

        # ------------------------------------------------------------------ #
        # 1. Parse incoming images
        # ------------------------------------------------------------------ #
        if bool(request.form) or (request.content_type and ("multipart" in request.content_type or "form" in request.content_type)):
            # ── File upload / Form post ──────────────────────────────────── #
            front_file = request.files.get("front") or request.files.get("image")
            back_file  = request.files.get("back")
            if not api_key:
                api_key = request.form.get("gemini_api_key")

            # Fallback: if only one image provided in 'back', treat it as front
            if (not front_file or not front_file.filename) and (back_file and back_file.filename):
                front_file = back_file
                back_file = None

            def _save_file(f) -> Path | None:
                if not f or not f.filename:
                    return None
                f.seek(0)
                ext = Path(f.filename).suffix or ".jpg"
                p = UPLOAD_DIR / f"{uuid.uuid4().hex}{ext}"
                f.save(str(p))
                return p

            barcode_file = request.files.get("barcode") or request.files.get("barcode_image")
            barcode_path = _save_file(barcode_file)
            barcode_number = request.form.get("barcode_number") or request.form.get("barcode")
            raw_h = request.form.get("package_height_mm", "")

            has_image = bool(front_file and front_file.filename)
            has_barcode = bool(barcode_number or barcode_path)

            if not has_image and not has_barcode:
                return jsonify({"error": "Please provide either a packaging label photo or a product barcode to analyze."}), 400

            front_path = _save_file(front_file)
            back_path  = _save_file(back_file)
            ruler_path = _save_file(request.files.get("ruler"))

        else:
            # ── JSON / base64 (camera capture) ───────────────────────────── #
            body = request.get_json(silent=True) or {}
            if not api_key:
                api_key = body.get("gemini_api_key")
            if body.get("engine"):
                engine = body["engine"]
            front_b64 = body.get("front_b64") or body.get("image_b64") or body.get("image")
            back_b64  = body.get("back_b64")
            ruler_b64 = body.get("ruler_b64")

            # Fallback: if only back_b64 provided, treat it as front
            if not front_b64 and back_b64:
                front_b64 = back_b64
                back_b64 = None

            def _save_b64(data_url: str | None) -> Path | None:
                if not data_url:
                    return None
                if "," in data_url:
                    data_url = data_url.split(",", 1)[1]
                p = UPLOAD_DIR / f"{uuid.uuid4().hex}.jpg"
                p.write_bytes(base64.b64decode(data_url))
                return p

            barcode_path = _save_b64(body.get("barcode_b64"))
            barcode_number = body.get("barcode_number") or body.get("barcode")
            raw_h = body.get("package_height_mm", "")

            has_image = bool(front_b64)
            has_barcode = bool(barcode_number or barcode_path)

            if not has_image and not has_barcode:
                return jsonify({"error": "Please provide either a packaging label photo or a product barcode to analyze."}), 400

            front_path = _save_b64(front_b64)
            back_path  = _save_b64(back_b64)
            ruler_path = _save_b64(ruler_b64)

        if raw_h:
            try:
                package_height_mm = float(raw_h)
            except ValueError:
                pass

        # ------------------------------------------------------------------ #
        # 2. Run the pipeline
        # ------------------------------------------------------------------ #
        report_path = str(REPORT_DIR / f"{uuid.uuid4().hex}.pdf")
        report = run_pipeline(
            front_image=str(front_path) if front_path else None,
            back_image=str(back_path)  if back_path  else None,
            ruler_image=str(ruler_path) if ruler_path else None,
            barcode_image=str(barcode_path) if barcode_path else None,
            barcode_number=barcode_number,
            package_height_mm=package_height_mm,
            output_path=report_path,
            ocr_engine=_ocr_engine,
            api_key=api_key,
            engine=engine,
        )

        # ------------------------------------------------------------------ #
        # 3. Serialise results
        # ------------------------------------------------------------------ #
        score = report.compliance_score
        rd    = report.rulebook_diff

        def serialise_results(results):
            return [
                {
                    "rule_id":   r.rule_id,
                    "rule_name": r.rule_name,
                    "status":    r.status,
                    "severity":  r.severity,
                    "detail":    r.detail,
                }
                for r in results
            ]

        fc = getattr(report.ebm_prediction, "feature_contributions", {}) or {}
        pd_obj = report.package_data
        if "barcode_registry_analysis" in fc:
            analysis_mode = "barcode_registry"
            srcs = ", ".join(getattr(pd_obj, "product_data_sources", []) or []) or "GS1 structural checks only"
            engine_desc = f"Bar Code / GS1 Registry Verification + LM(PC) Rules 2011 Rulebook — sources: {srcs}"
            if "llm_barcode_analysis" in fc:
                engine_desc += " (Gemini-enriched)"
        elif "visual_document_understanding" in fc:
            analysis_mode = "visual_ml"
            be = next((k.replace("backend_", "") for k in fc if k.startswith("backend_")), "model")
            geo = " + metric geometry" if "metric_geometry" in fc else ""
            engine_desc = f"Trained visual document model ({be}){geo} + LM(PC) Rules 2011 rulebook"
        elif "llm_barcode_analysis" in fc:
            analysis_mode = "llm_barcode"
            engine_desc = "Gemini LLM Barcode Verification & GS1 Registry"
        elif "llm_vision_analysis" in fc:
            analysis_mode = "llm_vision"
            engine_desc = "Gemini Multimodal Vision (Front & Back Label Photos)"
        else:
            analysis_mode = "local_ocr"
            engine_desc = "Local OCR & Rulebook Engine"

        payload = {
            "scan_id":   report.scan_id,
            "timestamp": report.scan_timestamp,
            "analysis_mode": analysis_mode,
            "engine_description": engine_desc,
            "score": {
                "final_score":   round(score.final_score * 100, 1),
                "star_rating":   score.star_rating,
                "star_label":    score.star_label,
                "ebm_score":     round(score.ebm_score * 100, 1),
                "rule_score":    round(score.rule_score * 100, 1),
                "passed_rules":  score.passed_rules,
                "failed_rules":  score.failed_rules,
            },
            "rules": {
                "passed":        serialise_results(rd.passed),
                "failed":        serialise_results(rd.failed),
                "warnings":      serialise_results(rd.warnings),
                "not_applicable":serialise_results(rd.not_applicable),
                "inconclusive":  serialise_results(rd.inconclusive),
            },
            "barcode": {
                "has_barcode":     pd_obj.has_barcode,
                "value":           pd_obj.barcode_value,
                "type":            pd_obj.barcode_type,
                "gtin_format":     pd_obj.barcode_gtin_format,
                "is_valid":        pd_obj.barcode_valid,
                "checksum_valid":  pd_obj.barcode_checksum_valid,
                "country":         pd_obj.barcode_country,
                "is_gs1_india":    pd_obj.barcode_is_gs1_india,
                "is_restricted":   pd_obj.barcode_is_restricted,
                "registered_owner": pd_obj.barcode_registered_owner,
            },
            "product": {
                "identified":   getattr(pd_obj, "product_identified", False),
                "sources":      getattr(pd_obj, "product_data_sources", []) or [],
                "name":         pd_obj.commodity_name,
                "manufacturer": pd_obj.manufacturer_name,
                "address":      pd_obj.manufacturer_address,
                "net_quantity": (f"{pd_obj.net_quantity_value} {pd_obj.net_quantity_unit}".strip()
                                 if pd_obj.net_quantity_value is not None else None),
                "mrp":          pd_obj.mrp_value,
                "country_of_origin": pd_obj.country_of_origin,
                "mfg_date":     pd_obj.manufacture_date,
                "fssai":        pd_obj.fssai_license_number,
                "provenance":   getattr(pd_obj, "data_provenance", {}) or {},
            },
            "recommendations": report.recommendations,
            "report_id": Path(report_path).stem,
            "images_used": {
                "front":   True,
                "back":    back_path  is not None,
                "ruler":   ruler_path is not None,
                "barcode": barcode_path is not None or barcode_number is not None,
            },
        }
        return jsonify(payload)

    except Exception as exc:
        logger.exception("Pipeline error")
        return jsonify({"error": str(exc), "traceback": traceback.format_exc()}), 500

    finally:
        for p in [front_path, back_path, ruler_path, barcode_path]:
            if p and p.exists():
                try:
                    p.unlink()
                except OSError:
                    pass


@app.route("/api/scan-barcode", methods=["POST"])
def scan_barcode_endpoint():
    """Decode a bar code / QR code (from an image, camera snapshot or a typed
    number) and structurally verify it as a GS1 GTIN.

    Pass ``lookup: true`` (JSON) or ``?lookup=1`` to additionally resolve the
    product through the public registries — skip it for the live camera loop so
    every frame stays fast.
    """
    from legal_metrology_ml.layer1_feature_extraction.barcode_scanner import BarcodeScanner
    from legal_metrology_ml.layer1_feature_extraction.gs1 import classify_gtin

    temp_path: Path | None = None
    try:
        body = {}
        do_lookup = request.args.get("lookup") in ("1", "true", "yes")
        manual_code = None

        if request.content_type and "multipart" in request.content_type:
            do_lookup = do_lookup or request.form.get("lookup") in ("1", "true", "yes")
            manual_code = (request.form.get("barcode_number") or "").strip() or None
            f = request.files.get("image") or request.files.get("file") or request.files.get("barcode")
            if f and f.filename:
                ext = Path(f.filename).suffix or ".jpg"
                temp_path = UPLOAD_DIR / f"bc_{uuid.uuid4().hex}{ext}"
                f.save(str(temp_path))
        else:
            body = request.get_json(force=True) or {}
            do_lookup = do_lookup or bool(body.get("lookup"))
            manual_code = (body.get("barcode_number") or body.get("code") or "").strip() or None
            data_url = body.get("image_b64") or body.get("barcode_b64") or body.get("image")
            if data_url:
                if "," in data_url:
                    data_url = data_url.split(",", 1)[1]
                temp_path = UPLOAD_DIR / f"bc_{uuid.uuid4().hex}.jpg"
                temp_path.write_bytes(base64.b64decode(data_url))

        barcodes = []
        if temp_path is not None:
            barcodes = BarcodeScanner().scan(temp_path)
        elif manual_code:
            from legal_metrology_ml.layer1_feature_extraction.barcode_scanner import BarcodeInfo
            gi = classify_gtin(manual_code)
            barcodes = [BarcodeInfo(
                code=gi.digits or manual_code,
                type=gi.fmt or "BARCODE",
                is_valid=gi.is_valid,
                country=gi.issuing_country,
                source="manual",
            )]
        else:
            return jsonify({"error": "Provide a bar code image or a barcode_number."}), 400

        results = []
        for b in barcodes:
            d = b.to_dict()
            gi = classify_gtin(b.code)
            d["gtin"] = gi.to_dict()
            d["is_valid"] = gi.is_valid if gi.is_gtin_length else d.get("is_valid")
            if gi.issuing_country and not d.get("country"):
                d["country"] = gi.issuing_country
            results.append(d)

        product = None
        if do_lookup and results:
            try:
                from legal_metrology_ml.data_sources.product_lookup import lookup_product
                rec = lookup_product(results[0]["code"])
                product = rec.to_dict()
            except Exception as e:  # noqa: BLE001
                logger.warning("Registry lookup failed in /api/scan-barcode: %s", e)

        return jsonify({
            "success": True,
            "count": len(results),
            "barcodes": results,
            "product": product,
        })
    except Exception as exc:
        logger.exception("Barcode scanning endpoint error")
        return jsonify({"error": str(exc), "traceback": traceback.format_exc()}), 500
    finally:
        if temp_path and temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


@app.route("/download/<report_id>")
def download_report(report_id: str):
    """Stream the generated PDF back to the browser."""
    # Sanitise: only hex characters allowed
    if not report_id.replace("-", "").isalnum():
        return "Invalid report ID", 400
    path = REPORT_DIR / f"{report_id}.pdf"
    if not path.exists():
        return "Report not found", 404
    return send_file(str(path), as_attachment=True, download_name="compliance_report.pdf")


@app.route("/api/settings/api-key", methods=["GET", "POST"])
def manage_api_key():
    """Get status of or update the Gemini API key."""
    from legal_metrology_ml.llm.llm_compliance_engine import get_api_key
    if request.method == "POST":
        data = request.get_json(force=True) if request.is_json else request.form
        key = (data.get("api_key") or "").strip()
        if key:
            os.environ["GEMINI_API_KEY"] = key
            env_path = Path(".env")
            lines = []
            if env_path.exists():
                lines = [l for l in env_path.read_text(encoding="utf-8").splitlines() if not l.startswith("GEMINI_API_KEY=") and not l.startswith("GOOGLE_API_KEY=")]
            lines.append(f"GEMINI_API_KEY={key}")
            env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            return jsonify({"success": True, "message": "Gemini API key saved successfully."})
        else:
            return jsonify({"error": "No API key provided."}), 400

    current = get_api_key()
    masked = (current[:6] + "..." + current[-4:]) if (current and len(current) > 10) else ("***" if current else "")
    return jsonify({
        "has_key": bool(current),
        "masked": masked,
    })


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n  Legal Metrology Compliance Checker")
    print("  ➜  Open http://localhost:5000 in your browser\n")
    app.run(host="0.0.0.0", port=5000, debug=True)
