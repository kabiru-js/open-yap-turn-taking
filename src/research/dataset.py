"""Read the Open Yap 1K sample from local tar shards.

Provides:
- ``load_all_metadata``: conversation-level metadata from the JSONs.
- ``iter_records``: yields (record, a_flac, b_flac) tuples by reading
  FLAC audio directly from tar via ``soundfile``.
- ``read_flac_from_tar``: read a single FLAC member as a numpy array.

All audio access is lazy (one conversation at a time) to minimise
disk footprint.
"""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

from .config import Config


# ── Tar helpers ──────────────────────────────────────────────────────────────

def _tar_paths(cfg: Config) -> List[Path]:
    tars = sorted(cfg.paths.raw_dir.glob("shard-*.tar"))
    if not tars:
        raise FileNotFoundError(
            f"No shard-*.tar files found in {cfg.paths.raw_dir}"
        )
    return tars


def _json_member_names(tar: tarfile.TarFile) -> List[str]:
    return sorted(m.name for m in tar.getmembers() if m.name.endswith(".json"))


def _flac_member_prefix(tar_name: str) -> str:
    return tar_name.rsplit("/", 1)[-1].replace(".json", "")


# ── Public API ───────────────────────────────────────────────────────────────

def load_all_metadata(cfg: Config) -> Dict[str, dict]:
    """Load JSON metadata for every conversation in the sample.

    Returns
    -------
    dict
        ``{conversation_id: metadata_dict}``
    """
    meta: Dict[str, dict] = {}
    for tar_path in _tar_paths(cfg):
        with tarfile.open(str(tar_path)) as tar:
            for jm in _json_member_names(tar):
                data = json.loads(tar.extractfile(jm).read())
                meta[data["conversation_id"]] = data
    return meta


def read_flac_from_tar(
    tar: tarfile.TarFile,
    member_name: str,
    target_sr: Optional[int] = None,
) -> Tuple[np.ndarray, int]:
    """Read a FLAC member from an open tar as a float32 numpy array.

    Parameters
    ----------
    tar : tarfile.TarFile
        An already-opened tar archive.
    member_name : str
        The full member path ending in ``.flac``.
    target_sr : int, optional
        If provided, resample to this sample rate.

    Returns
    -------
    audio : np.ndarray, shape ``(n_samples,)`` or ``(n_samples, n_channels)``
    sr : int
        Actual sample rate of the returned audio.
    """
    import soundfile as sf

    fobj = tar.extractfile(member_name)
    if fobj is None:
        raise KeyError(f"Member {member_name} not found in tar")
    buf = io.BytesIO(fobj.read())
    audio, sr = sf.read(buf, dtype="float32", always_2d=False)
    audio = audio.squeeze()
    if target_sr is not None and sr != target_sr:
        from scipy.signal import resample_poly
        import math
        g = math.gcd(sr, target_sr)
        audio = resample_poly(audio, target_sr // g, sr // g).astype(np.float32)
        sr = target_sr
    return audio, sr


def iter_records(
    cfg: Config,
) -> Iterator[Tuple[dict, np.ndarray, np.ndarray, str]]:
    """Iterate over all conversations yielding (metadata, audio_a, audio_b, conv_id).

    Audio is loaded at ``cfg.audio.sample_rate`` (default 48 kHz). The tar
    archive for each shard is opened and closed internally; callers only
    receive the decoded data.
    """
    sr = cfg.audio.sample_rate
    for tar_path in _tar_paths(cfg):
        with tarfile.open(str(tar_path)) as tar:
            for jm in _json_member_names(tar):
                data = json.loads(tar.extractfile(jm).read())
                conv_id = data["conversation_id"]
                prefix = _flac_member_prefix(jm)
                a_name = f"{prefix}.a.flac"
                b_name = f"{prefix}.b.flac"
                a_audio, a_sr = read_flac_from_tar(tar, a_name, target_sr=sr)
                b_audio, b_sr = read_flac_from_tar(tar, b_name, target_sr=sr)
                # Sanity: timeline alignment
                assert a_audio.shape == b_audio.shape, (
                    f"Channel lengths differ for {conv_id}: "
                    f"{a_audio.shape} vs {b_audio.shape}"
                )
                yield data, a_audio, b_audio, conv_id


def count_conversations(cfg: Config) -> int:
    n = 0
    for tar_path in _tar_paths(cfg):
        with tarfile.open(str(tar_path)) as tar:
            n += len(_json_member_names(tar))
    return n


def build_speaker_map(metadata: Dict[str, dict]) -> Dict[str, dict]:
    """Build a flat map of speaker_id -> info across all conversations."""
    speakers: Dict[str, dict] = {}
    for conv_id, rec in metadata.items():
        for side in ("a", "b"):
            spk = rec["speakers"][side]
            sid = spk["speaker_id"]
            if sid not in speakers:
                speakers[sid] = {
                    "speaker_id": sid,
                    "conversations": [],
                    "age_range": spk.get("age_range"),
                    "gender": spk.get("gender"),
                    "country": spk.get("country"),
                }
            speakers[sid]["conversations"].append(
                {"conversation_id": conv_id, "side": side}
            )
    return speakers
