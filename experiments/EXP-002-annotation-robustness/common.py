"""Shared helpers for EXP-002: annotation/segmentation corruption robustness.

Everything reads only *cached* artifacts (``data/interim`` frame features,
decision frames, dataset) plus the raw tar JSON metadata (once, for the
per-conversation noise-floor metrics used by the original VAD). No audio is
decoded in any experiment run.

All model code mirrors ``scripts/05_models.py`` exactly (same hyperparameters,
same 90000-row cap with ``random_state=seed``, same median-impute, same
StandardScaler for LR). The σ = 0 (Arm A) and margin = 0 (Arm B) guards guarantee
the perturbation machinery is a faithful superset of the published EXP-001
pipeline.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from research.config import Config
from research.events import build_decision_frame, build_segments
from research.window_features import FEAT_COLS, build_window_features

EXP_DIR = Path(__file__).resolve().parent
RESULT_DIR = EXP_DIR / "results"
FIGURE_DIR = EXP_DIR / "figures"
CONFIG_PATH = EXP_DIR / "config.json"
_METRICS_CACHE = EXP_DIR / "noise_floor_metrics.json"


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text())


def frame_idx_of_time(df: pd.DataFrame) -> pd.Series:
    """Recover the 10 ms frame index for decision points (grid multiples of 20)."""
    return (df["time_s"] / 0.01).round().astype(np.int64)


# ── Arm A: labels from perturbed onsets ──────────────────────────────────────

def load_decision_rows(cfg: Config) -> pd.DataFrame:
    """Decision-level rows with onset times and stored labels (all horizons)."""
    ev = pd.read_parquet(cfg.paths.interim_dir / "events" / "all_decisions.parquet")
    ev["conv_id"] = ev["conv_id"].astype(str)
    ev["frame_idx"] = ev["frame_idx"].astype(np.int64)
    return ev.reset_index(drop=True)


def load_dataset_frame_idx(cfg: Config) -> pd.DataFrame:
    df = pd.read_parquet(cfg.paths.interim_dir / "dataset.parquet")
    df["conv_id"] = df["conv_id"].astype(str)
    df["frame_idx"] = frame_idx_of_time(df).astype(np.int64)
    return df


def merge_onsets(cfg: Config) -> pd.DataFrame:
    """Join the ML dataset to decision rows (onsets + stored labels), 1:1.

    Both tables derive from the same decision frame in ``02_define_events.py``
    after identical tail exclusion, so the join on (conv_id, frame_idx) must be
    complete.
    """
    dset = load_dataset_frame_idx(cfg)
    ev = load_decision_rows(cfg)
    keys = ["conv_id", "frame_idx"]
    joined = dset.merge(
        ev[keys + ["next_other_onset_s"]],
        on=keys,
        how="left",
        validate="one_to_one",
    )
    assert joined["next_other_onset_s"].notna().all(), "onset join incomplete"
    return joined


def perturb_onsets(onsets: np.ndarray, kind: str, sigma_ms: float,
                   shift_ms: float, rng: np.random.Generator) -> np.ndarray:
    """Return a perturbed copy of the per-decision onset array (seconds).

    - clean   : unchanged
    - jitter  : additive Gaussian noise, onset clipped to >= 0
    - shift   : constant additive offset (both signs)
    - shuffle : GLOBAL permutation of the finite onsets across the whole array
      (NOT the reported control). A global onset permutation re-introduces a
      time profile into the label (P(label|t) grows with the t+h threshold)
      that the ``elapsed_s`` feature can exploit, inflating ROC to ~0.77.
      The reported chance control is a within-conversation permutation of the
      *labels* (see ``run_arm_a_controls_lowmem.py``).
    ``np.inf`` (no new onset within the recording tail) is preserved.
    """
    ons = onsets.astype(np.float64).copy()
    finite = np.isfinite(ons)
    vals = ons[finite]
    if kind == "clean":
        pass
    elif kind == "jitter":
        vals = np.maximum(vals + rng.normal(0.0, sigma_ms / 1000.0, size=len(vals)), 0.0)
    elif kind == "shift":
        vals = np.maximum(vals + shift_ms / 1000.0, 0.0)
    elif kind == "shuffle":
        vals = rng.permutation(vals)
    else:
        raise ValueError(f"unknown kind {kind!r}")
    ons[finite] = vals
    return ons


def labels_from_onsets(time_s: np.ndarray, onsets: np.ndarray, h_ms: int) -> np.ndarray:
    """Recompute ``label_h = (onset <= t + h)`` exactly as events.py does."""
    return (onsets <= (time_s + h_ms / 1000.0)).astype(np.int8)


# ── Shared LOCO evaluation (mirrors scripts/05_models.py) ────────────────────

def _safe_roc(y, p):
    if len(np.unique(y)) < 2:
        return np.nan
    return roc_auc_score(y, p)


def _safe_pr(y, p):
    if len(np.unique(y)) < 2:
        return np.nan
    return average_precision_score(y, p)


def eval_loco(df: pd.DataFrame, ycol: str, models=("lr", "gbdt"),
              seed: int = 42, cap: int = 90000) -> dict:
    """Leave-one-conversation-out evaluation for one target column.

    Returns ``{"lr_<label>": {...}, "gbdt_<label>": {...}}`` with per-fold
    ROC/PR lists plus summarized metrics.
    """
    groups = sorted(df["conv_id"].astype(str).unique())
    res = {m: {"roc": [], "pr": [], "n_test": [], "groups": []} for m in models}

    t0 = time.time()
    for i, grp in enumerate(groups):
        fl_tr = df[df["conv_id"].astype(str) != grp]
        fl_te = df[df["conv_id"].astype(str) == grp]
        if len(fl_te) == 0:
            continue
        if len(fl_tr) > cap:
            fl_tr = fl_tr.sample(cap, random_state=seed)
        imputer = SimpleImputer(strategy="median")
        Xi = imputer.fit_transform(fl_tr[FEAT_COLS])
        Xe = imputer.transform(fl_te[FEAT_COLS])
        ytr = fl_tr[ycol].to_numpy()
        yte = fl_te[ycol].to_numpy()

        if "lr" in models:
            sc = StandardScaler().fit(Xi)
            clf = LogisticRegression(max_iter=2000, C=1.0)
            clf.fit(sc.transform(Xi), ytr)
            p = clf.predict_proba(sc.transform(Xe))[:, 1]
            res["lr"]["roc"].append(_safe_roc(yte, p))
            res["lr"]["pr"].append(_safe_pr(yte, p))
            res["lr"]["n_test"].append(len(fl_te))
            res["lr"]["groups"].append(grp)

        if "gbdt" in models:
            clf = HistGradientBoostingClassifier(
                max_iter=160, learning_rate=0.12, random_state=seed)
            clf.fit(Xi, ytr)
            p = clf.predict_proba(Xe)[:, 1]
            res["gbdt"]["roc"].append(_safe_roc(yte, p))
            res["gbdt"]["pr"].append(_safe_pr(yte, p))
            res["gbdt"]["n_test"].append(len(fl_te))
            res["gbdt"]["groups"].append(grp)
        print(f"    fold {grp} ({i + 1}/{len(groups)}) done {time.time() - t0:.0f}s",
              flush=True)
    return res


def summarize_eval(res: dict) -> dict:
    """Convert per-fold lists for each model into summary dicts."""
    out = {}
    for name, r in res.items():
        rocs = np.asarray(r["roc"], dtype=float)
        prs = np.asarray(r["pr"], dtype=float)
        wt = np.asarray(r["n_test"])
        finite = np.isfinite(rocs)
        out[name] = {
            "roc_auc": float(np.nanmean(rocs)) if finite.any() else None,
            "roc_auc_sd": float(np.nanstd(rocs)) if finite.any() else None,
            "pr_auc": float(np.nanmean(prs)) if finite.any() else None,
            "pr_auc_sd": float(np.nanstd(prs)) if finite.any() else None,
            "pr_auc_weighted": float(np.nansum(prs * wt) / wt.sum()) if wt.sum() else None,
            "n_folds_ok": int(finite.sum()),
            "per_fold_roc": [None if np.isnan(x) else float(x) for x in rocs],
            "per_fold_pr": [None if np.isnan(x) else float(x) for x in prs],
            "n_test": [int(x) for x in wt],
            "groups": list(r["groups"]),
        }
    return out


# ── Fast path: precompute frozen folds (features unchanged across Arm A) ────

def prep_loco_folds(df: pd.DataFrame, seed: int = 42, cap: int = 90000) -> list:
    """Precompute per-fold sample splits, imputed matrices once (Arm A only).

    Returns ``[{"grp": str, "tr_idx": ndarray, "te_idx": ndarray,
                "Xi": ndarray, "Xe": ndarray}, ...]``.
    """
    conv_str = df["conv_id"].astype(str).to_numpy()
    groups = sorted(set(conv_str))
    folds = []
    for grp in groups:
        te_idx = np.flatnonzero(conv_str == grp)
        if len(te_idx) == 0:
            continue
        tr_idx = np.flatnonzero(conv_str != grp)
        if len(tr_idx) > cap:
            rs = np.random.RandomState(seed)
            pick = rs.choice(len(tr_idx), size=cap, replace=False)
            tr_idx = tr_idx[pick]
        imputer = SimpleImputer(strategy="median")
        Xi = imputer.fit_transform(df.iloc[tr_idx][FEAT_COLS])
        Xe = imputer.transform(df.iloc[te_idx][FEAT_COLS])
        folds.append({"grp": grp, "tr_idx": tr_idx, "te_idx": te_idx,
                       "Xi": Xi, "Xe": Xe})
    return folds


def eval_loco_fast(folds: list, y: np.ndarray,
                   models: list = ("lr",), seed: int = 42) -> dict:
    """LOCO eval using precomputed folds + externally-supplied label array.

    ``y`` must align row-by-row with the original dataset used to build
    ``folds`` (same order, same length).
    """
    res = {m: {"roc": [], "pr": [], "n_test": [], "groups": []} for m in models}
    for f in folds:
        ytr = y[f["tr_idx"]]
        yte = y[f["te_idx"]]
        if "lr" in models:
            sc = StandardScaler().fit(f["Xi"])
            clf = LogisticRegression(max_iter=2000, C=1.0)
            clf.fit(sc.transform(f["Xi"]), ytr)
            p = clf.predict_proba(sc.transform(f["Xe"]))[:, 1]
            res["lr"]["roc"].append(_safe_roc(yte, p))
            res["lr"]["pr"].append(_safe_pr(yte, p))
            res["lr"]["n_test"].append(len(yte))
            res["lr"]["groups"].append(f["grp"])
        if "gbdt" in models:
            clf = HistGradientBoostingClassifier(
                max_iter=160, learning_rate=0.12, random_state=seed)
            clf.fit(f["Xi"], ytr)
            p = clf.predict_proba(f["Xe"])[:, 1]
            res["gbdt"]["roc"].append(_safe_roc(yte, p))
            res["gbdt"]["pr"].append(_safe_pr(yte, p))
            res["gbdt"]["n_test"].append(len(yte))
            res["gbdt"]["groups"].append(f["grp"])
    return res


# ── Arm B: VAD re-derivation from cached RMS (mirrors research/audio.py) ─────

def load_noise_floor_metrics(cfg: Config) -> dict:
    """Per-conversation {a: dB, b: dB} noise-floor metrics used by the VAD.

    Extracted once from the raw tar JSON metadata (no audio access) and cached
    in this experiment directory as a tiny JSON.
    """
    if _METRICS_CACHE.exists():
        return json.loads(_METRICS_CACHE.read_text())
    from research.dataset import load_all_metadata
    meta = load_all_metadata(cfg)
    out = {}
    for cid, m in meta.items():
        out[cid] = {
            "a": float(m["speakers"]["a"]["audio_metrics"].get("noise_floor_dbfs") or -70.0),
            "b": float(m["speakers"]["b"]["audio_metrics"].get("noise_floor_dbfs") or -70.0),
        }
    _METRICS_CACHE.write_text(json.dumps(out, indent=2, sort_keys=True))
    print(f"[metrics] cached {len(out)} conversations -> {_METRICS_CACHE}", flush=True)
    return out


def _apply_hangover(mask: np.ndarray, hangover: int, min_speech_frames: int) -> np.ndarray:
    """Identical smoothing as ``research.audio.compute_vad`` (lines 239-269)."""
    result = mask.copy()
    in_speech, count = False, 0
    for i in range(len(result)):
        if result[i]:
            in_speech, count = True, 0
        else:
            if in_speech:
                count += 1
                if count <= hangover:
                    result[i] = True
                else:
                    in_speech, count = False, 0
    in_speech, seg_start = False, 0
    for i in range(len(result) + 1):
        if i < len(result) and result[i]:
            if not in_speech:
                seg_start = i
                in_speech = True
        else:
            if in_speech:
                if i - seg_start < min_speech_frames:
                    result[seg_start:i] = False
                in_speech = False
    return result


def recompute_vad_from_rms(rms_a: np.ndarray, rms_b: np.ndarray,
                           nf_a: float, nf_b: float,
                           margin_db: float, cfg: Config):
    """Re-derive VAD masks from cached RMS frames at an absolute margin.

    ``margin_db`` is the *absolute* speech threshold above the noise floor
    (EXP-001 used ``cfg.features.vad_margin_above_noise_db = 20``; perturbed
    margins are 20 + delta).
    """
    thr_a = max(nf_a, float(np.percentile(rms_a, 25))) + margin_db
    thr_b = max(nf_b, float(np.percentile(rms_b, 25))) + margin_db
    raw_a = rms_a > thr_a
    raw_b = rms_b > thr_b
    hangover = cfg.audio.hangover_frames
    min_speech_frames = int(cfg.audio.min_speech_ms / cfg.audio.hop_ms)
    return (
        _apply_hangover(raw_a, hangover, min_speech_frames),
        _apply_hangover(raw_b, hangover, min_speech_frames),
    )


def build_dataset_at_margin(cfg: Config, metrics: dict, margin_db: float,
                            horizons_ms, window_frames: int) -> pd.DataFrame:
    """Full re-derivation: cached rms -> VAD -> segments -> decisions -> features.

    With ``margin_db = 20`` this reproduces the stored ``dataset.parquet``
    bit-for-bit (asserted by ``tests/test_exp002.py``).
    """
    feat_dir = cfg.paths.interim_dir / "features"
    parts = []
    t0 = time.time()
    for p in sorted(feat_dir.glob("conv_*.npz")):
        cid = p.stem
        npz = np.load(p, allow_pickle=True)
        n_frames = len(npz["vad_a"])
        rms = {s: npz[f"{s}_rms_db"] for s in ("a", "b")}
        vad_a, vad_b = recompute_vad_from_rms(rms["a"], rms["b"],
                                              metrics[cid]["a"], metrics[cid]["b"],
                                              margin_db, cfg)
        segments = build_segments(vad_a, vad_b, cfg.audio.hop_ms,
                                  max_gap_ms=cfg.audio.max_gap_within_speaker_ms)
        decisions = build_decision_frame(
            cid, segments, n_frames,
            hop_ms=cfg.audio.hop_ms,
            decision_grid_ms=cfg.prediction.decision_grid_ms,
            horizons_ms=horizons_ms,
        )
        decisions["current_speaker"] = decisions["current_speaker"].astype("category")
        decisions["other_speaker"] = decisions["other_speaker"].astype("category")
        wf = build_window_features(npz, decisions, window_frames)
        for h in horizons_ms:
            wf[f"label_{h}"] = decisions[f"label_{h}"].to_numpy()
        parts.append(wf)
        print(f"    {cid}  rows={len(wf)}  ({time.time() - t0:.0f}s)", flush=True)
        del npz
    df = pd.concat(parts, ignore_index=True)
    df["conv_id"] = df["conv_id"].astype("category")
    return df


# ── Persistence ──────────────────────────────────────────────────────────────

def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


def stamp_meta() -> dict:
    return {
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pipeline": "EXP-002 annotation/segmentation corruption robustness",
        "train_cap": 90000,
        "train_seed": 42,
    }