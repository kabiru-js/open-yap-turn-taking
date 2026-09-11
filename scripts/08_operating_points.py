"""Threshold operating points from pooled GBDT predictions @1000ms."""
import sys
sys.path.insert(0, "src")
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve
from research.config import Config

cfg = Config()
d = np.load(cfg.paths.results_dir / "predictions_1000.npz")
y, pg = d["y"], d["p_gb"]
base = y.mean()
pr, rc, th = precision_recall_curve(y, pg)
# best F1 threshold
f1 = 2 * pr * rc / (pr + rc + 1e-9)
best = np.argmax(f1)
for t in (0.2, 0.3, 0.5, 0.7):
    p = (pg >= t).mean()
    tp = (pg[y == 1] >= t).mean() if (y == 1).any() else 0
    pr_t = (pg[y == 1] >= t).sum()
    pred_t = (pg >= t).sum()
    prec = (pg[(pg >= t) & (y == 1)].size / max(pred_t, 1))
    print(f"thr={t:.2f} pred_rate={p:.4f} recall={tp:.4f} precision={prec:.4f}")
print(f"best F1 thr={th[best]:.3f} prec={pr[best]:.3f} rec={rc[best]:.3f} f1={f1[best]:.3f}")
print(f"base rate={base:.3f}")