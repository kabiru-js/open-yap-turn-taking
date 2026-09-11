"""Typed configuration for turn-taking experiments.

All experiment parameters are defined here. Default values come from
``configs/turn_prediction.yaml``; anything not in the YAML falls back
to these defaults.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Paths:
    data_dir: Path = ROOT / "data"
    raw_dir: Path = ROOT / "data" / "raw"
    json_dir: Path = ROOT / "data" / "raw" / "json"
    interim_dir: Path = ROOT / "data" / "interim"
    results_dir: Path = ROOT / "results"
    figures_dir: Path = ROOT / "figures"
    figures_raw: Path = ROOT / "figures" / "raw"


@dataclass
class Audio:
    sample_rate: int = 48000
    frame_len_ms: float = 30.0
    hop_ms: float = 10.0
    min_energy_db: float = -40.0
    min_speech_ms: float = 200.0
    hangover_frames: int = 6
    max_gap_within_speaker_ms: float = 400.0


@dataclass
class Features:
    vad_margin_above_noise_db: float = 20.0
    f0_range: tuple = (60, 400)
    spectral_n_fft: int = 2048
    spectral_hop: int = 512


@dataclass
class Events:
    # turn taxonomy thresholds
    clean_gap_min_ms: float = 0.0
    clean_gap_max_ms: float = 1200.0
    overlap_min_dur_ms: float = 30.0
    interruption_post_offset_ms: float = 800.0
    backchannel_max_dur_ms: float = 800.0
    backchannel_max_words: int = 3


@dataclass
class Prediction:
    horizons_ms: List[int] = field(default_factory=lambda: [200, 500, 750, 1000, 1500])
    decision_grid_ms: int = 200
    lookback_s: float = 1.5
    floor_lookback_s: float = 2.0


@dataclass
class Model:
    n_splits_loco: int = 16
    test_conv_ids: List[str] = field(default_factory=list)
    seed: int = 42


@dataclass
class Config:
    paths: Paths = field(default_factory=Paths)
    audio: Audio = field(default_factory=Audio)
    features: Features = field(default_factory=Features)
    events: Events = field(default_factory=Events)
    prediction: Prediction = field(default_factory=Prediction)
    model: Model = field(default_factory=Model)
