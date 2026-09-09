"""Geometry-aware augmentation for training + the E6 robustness study.

Every transform updates the region boxes so annotations stay valid. Pure
numpy/OpenCV — no extra deps.

    aug = Augmenter(seed=0)
    img2, ann2 = aug(img, ann)                       # random pipeline
    img2, ann2 = aug.apply(img, ann, ["perspective", "glare", "motion_blur"])
"""
from __future__ import annotations

import random

import cv2
import numpy as np

from ml.data.schema import BBox, ImageAnnotation, Region

_ALL = ["perspective", "rotate", "motion_blur", "defocus", "glare", "lowlight",
        "brightness_contrast", "gauss_noise", "jpeg"]


class Augmenter:
    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)
        self.np = np.random.default_rng(seed)

    def __call__(self, img: np.ndarray, ann: ImageAnnotation):
        k = self.rng.randint(1, 3)
        ops = self.rng.sample(_ALL, k)
        return self.apply(img, ann, ops)

    def apply(self, img: np.ndarray, ann: ImageAnnotation, ops: list[str]):
        img = img.copy()
        regions = [r.model_copy(deep=True) for r in ann.regions]
        for op in ops:
            img, regions = getattr(self, f"_{op}")(img, regions)
        new = ann.model_copy(deep=True)
        new.height, new.width = img.shape[:2]
        new.regions = [r for r in regions if r.bbox.area > 1e-5]
        new.annotator_notes = f"{ann.annotator_notes or ''} | aug:{'+'.join(ops)}"
        return img, new

    # ── helpers ───────────────────────────────────────────────────────────
    @staticmethod
    def _warp_boxes(regions, M, w, h):
        out = []
        for r in regions:
            b = r.bbox
            pts = np.array([[b.x0 * w, b.y0 * h], [b.x1 * w, b.y0 * h],
                            [b.x1 * w, b.y1 * h], [b.x0 * w, b.y1 * h]], dtype=np.float32)
            pts = cv2.perspectiveTransform(pts[None], M)[0]
            x0, y0 = pts.min(0)
            x1, y1 = pts.max(0)
            r.bbox = BBox(
                x0=float(np.clip(x0 / w, 0, 1)), y0=float(np.clip(y0 / h, 0, 1)),
                x1=float(np.clip(x1 / w, 0, 1)), y1=float(np.clip(y1 / h, 0, 1)),
            )
            out.append(r)
        return out

    # ── geometric ─────────────────────────────────────────────────────────
    def _perspective(self, img, regions):
        h, w = img.shape[:2]
        d = self.np.uniform(0.02, 0.12)
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        off = (self.np.uniform(-d, d, (4, 2)) * [w, h]).astype(np.float32)
        M = cv2.getPerspectiveTransform(src, src + off)
        return cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE), \
            self._warp_boxes(regions, M, w, h)

    def _rotate(self, img, regions):
        h, w = img.shape[:2]
        a = self.np.uniform(-8, 8)
        M2 = cv2.getRotationMatrix2D((w / 2, h / 2), a, 1.0)
        M = np.vstack([M2, [0, 0, 1]]).astype(np.float32)
        return cv2.warpAffine(img, M2, (w, h), borderMode=cv2.BORDER_REPLICATE), \
            self._warp_boxes(regions, M, w, h)

    # ── photometric (boxes unchanged) ─────────────────────────────────────
    def _motion_blur(self, img, regions):
        k = int(self.np.integers(7, 21)) | 1
        ker = np.zeros((k, k), np.float32)
        ker[k // 2, :] = 1.0 / k
        ang = self.np.uniform(0, 180)
        ker = cv2.warpAffine(ker, cv2.getRotationMatrix2D((k / 2, k / 2), ang, 1), (k, k))
        return cv2.filter2D(img, -1, ker), regions

    def _defocus(self, img, regions):
        k = int(self.np.integers(3, 11)) | 1
        return cv2.GaussianBlur(img, (k, k), 0), regions

    def _glare(self, img, regions):
        h, w = img.shape[:2]
        cx, cy = self.np.uniform(0.2, 0.8, 2) * [w, h]
        rad = self.np.uniform(0.15, 0.4) * min(w, h)
        Y, X = np.ogrid[:h, :w]
        mask = np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * rad ** 2))
        out = img.astype(np.float32) + (mask[..., None] * self.np.uniform(80, 180))
        return np.clip(out, 0, 255).astype(np.uint8), regions

    def _lowlight(self, img, regions):
        g = self.np.uniform(0.25, 0.55)
        return np.clip(img.astype(np.float32) * g, 0, 255).astype(np.uint8), regions

    def _brightness_contrast(self, img, regions):
        a = self.np.uniform(0.7, 1.3)
        b = self.np.uniform(-30, 30)
        return np.clip(img.astype(np.float32) * a + b, 0, 255).astype(np.uint8), regions

    def _gauss_noise(self, img, regions):
        n = self.np.normal(0, self.np.uniform(4, 18), img.shape)
        return np.clip(img.astype(np.float32) + n, 0, 255).astype(np.uint8), regions

    def _jpeg(self, img, regions):
        q = int(self.np.integers(25, 70))
        ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
        return (cv2.imdecode(enc, cv2.IMREAD_COLOR) if ok else img), regions


def make_robustness_set(root: str, split: str, out: str, per_image: int = 3, seed: int = 0) -> None:
    """Write perturbed copies of a split for the E6 robustness evaluation."""
    from pathlib import Path

    from ml.data.dataset import PcrLabelDataset

    ds = PcrLabelDataset(root, split=split)
    outp = Path(out)
    (outp / "images").mkdir(parents=True, exist_ok=True)
    (outp / "annotations").mkdir(parents=True, exist_ok=True)
    aug = Augmenter(seed)
    for ann in ds:
        img = cv2.imread(str(ds.image_path(ann)))
        if img is None:
            continue
        for j in range(per_image):
            im2, an2 = aug(img, ann)
            stem = f"{Path(ann.image).stem}_aug{j}"
            an2.image = f"images/{stem}.jpg"
            an2.split = "robust"
            cv2.imwrite(str(outp / "images" / f"{stem}.jpg"), im2)
            (outp / "annotations" / f"{stem}.json").write_text(an2.model_dump_json(indent=2))
    PcrLabelDataset(out).save_index()
    print(f"robustness set -> {out}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", default="gold")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-image", type=int, default=3)
    a = ap.parse_args()
    make_robustness_set(a.root, a.split, a.out, a.per_image)
