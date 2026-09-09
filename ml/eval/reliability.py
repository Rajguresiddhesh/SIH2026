"""Calibration diagnostics for the soft compliance scores (paper §6 E4).

- Expected Calibration Error (ECE) and Maximum CE (MCE).
- Reliability diagram (matplotlib, saved to PNG).
- Selective-risk / coverage curve for the abstaining classifier.
- Temperature scaling to recalibrate a miscalibrated score.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _bin_stats(p: np.ndarray, y: np.ndarray, n_bins: int):
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    conf, acc, cnt = np.zeros(n_bins), np.zeros(n_bins), np.zeros(n_bins)
    for b in range(n_bins):
        m = idx == b
        if m.any():
            conf[b] = p[m].mean()
            acc[b] = y[m].mean()
            cnt[b] = m.sum()
    return edges, conf, acc, cnt


def ece(p_fail, y_fail, n_bins: int = 10) -> dict:
    """p_fail = predicted P(FAIL); y_fail = 1 if gold is FAIL."""
    p = np.asarray(p_fail, dtype=float)
    y = np.asarray(y_fail, dtype=float)
    _, conf, acc, cnt = _bin_stats(p, y, n_bins)
    w = cnt / max(cnt.sum(), 1)
    gap = np.abs(conf - acc)
    return {"ECE": float((w * gap).sum()), "MCE": float(gap[cnt > 0].max() if (cnt > 0).any() else 0.0),
            "n": int(cnt.sum())}


def temperature_scale(logits, y, iters: int = 200, lr: float = 0.05) -> float:
    """Fit a scalar T minimising NLL of sigmoid(logit / T). Returns T."""
    z = np.asarray(logits, dtype=float)
    y = np.asarray(y, dtype=float)
    T = 1.0
    for _ in range(iters):
        s = 1.0 / (1.0 + np.exp(-z / T))
        # d NLL / d T
        g = np.mean((s - y) * (-z / T**2) * s * (1 - s) / np.clip(s * (1 - s), 1e-6, None))
        T = max(0.05, T - lr * g)
    return float(T)


def reliability_diagram(p_fail, y_fail, out_png: str, n_bins: int = 10, title: str = "") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = np.asarray(p_fail, float)
    y = np.asarray(y_fail, float)
    edges, conf, acc, cnt = _bin_stats(p, y, n_bins)
    centres = (edges[:-1] + edges[1:]) / 2
    e = ece(p, y, n_bins)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax.bar(centres, acc, width=1 / n_bins, edgecolor="k", alpha=0.75, label="accuracy")
    ax.plot(centres[cnt > 0], conf[cnt > 0], "o-", color="crimson", label="confidence")
    ax.set_xlabel("predicted P(FAIL)")
    ax.set_ylabel("empirical FAIL rate")
    ax.set_title(f"{title}  ECE={e['ECE']:.3f}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def selective_risk_curve(p_correct, out_png: str | None = None) -> list[tuple[float, float]]:
    """p_correct = model's confidence that each covered prediction is right
    (here 2*|p_fail-0.5|). Returns [(coverage, risk)] sorted by threshold."""
    c = np.sort(np.asarray(p_correct, float))[::-1]
    n = len(c)
    pts = []
    correct = np.asarray(p_correct, float) >= 0.5  # placeholder if labels not given
    for k in range(1, n + 1):
        thr = c[k - 1]
        keep = np.asarray(p_correct, float) >= thr
        cov = keep.mean()
        risk = 1.0 - correct[keep].mean() if keep.any() else 0.0
        pts.append((float(cov), float(risk)))
    if out_png:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        xs, ys = zip(*pts)
        plt.figure(figsize=(4, 3))
        plt.plot(xs, ys, "-")
        plt.xlabel("coverage")
        plt.ylabel("selective risk")
        plt.tight_layout()
        Path(out_png).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_png, dpi=140)
        plt.close()
    return pts


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500).astype(float)
    # a slightly over-confident score
    p = np.clip(y * 0.7 + 0.15 + rng.normal(0, 0.18, 500), 0, 1)
    print(json.dumps(ece(p, y), indent=2))
    reliability_diagram(p, y, "runs/reliability_demo.png", title="demo")
    print("wrote runs/reliability_demo.png")
