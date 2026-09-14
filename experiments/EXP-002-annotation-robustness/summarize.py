import json
from pathlib import Path

base = Path("experiments/EXP-002-annotation-robustness")

r = json.loads((base / "results/exp002_arm_b.json").read_text())
print("=== Arm B ===")
for tag, d in sorted(r["datasets"].items()):
    cons = d["cells"].get("consistent", {}).get("lr", {})
    print(f'{tag}: base={d["base_rate"]:.4f} n={d["n_decision_points"]}  '
          f'consistent LR ROC={cons.get("roc_auc", float("nan")):.4f} '
          f'PR={cons.get("pr_auc", float("nan")):.4f}')
    dr = d["cells"].get("drift_labels_from_clean", {}).get("lr", {})
    if dr:
        print(f'      drift(clean labels) LR ROC={dr.get("roc_auc", float("nan")):.4f} '
              f'PR={dr.get("pr_auc", float("nan")):.4f} '
              f'({d["cells"]["drift_labels_from_clean"].get("n_join_rows")} rows)')

print("\n=== Arm A ===")
a = json.loads((base / "results/exp002_arm_a.json").read_text())
rows = []
for name, c in sorted(a["conditions"].items()):
    ev = c["evals"].get("label_1000", {})
    lr = ev.get("lr", {})
    rows.append((name, (ev.get("base_rate"), lr.get("roc_auc"), lr.get("pr_auc"))))

order = ["clean"] + [f"jitter_s101_sigma{s}" for s in (0, 25, 50, 100, 200, 400)] \
    + [f"jitter_s202_sigma{s}" for s in (0, 25, 100, 200)] \
    + [f"jitter_s303_sigma{s}" for s in (0, 25, 100)] \
    + ["shift_m200", "shift_p200", "shuffle", "uniform"]
by_name = {n: v for n, v, *_ in rows}
for n in order:
    if n in by_name:
        br, roc, pr = by_name[n]
        print(f'  {n:24s} base={br:.4f} ROC={roc:.4f} PR={pr:.4f}')