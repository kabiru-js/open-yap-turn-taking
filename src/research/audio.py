"""Audio feature extraction — VAD, frame-level features, pitch tracking.

All features are computed on a 10ms hop at 48 kHz using numpy/scipy only.
"""

from __future__ import annotations
from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from .config import Config


# ── Frame-level helpers ──────────────────────────────────────────────────────

def rms_frames(signal: NDArray, frame_len: int, hop: int) -> NDArray:
    """RMS energy per frame in dBFS (zero-copy framing, no 2-D array)."""
    n_pad = frame_len - len(signal) % hop
    sig = np.pad(signal, (0, n_pad), mode="constant")
    cum = np.cumsum(sig ** 2)
    ends = np.arange(frame_len - 1, len(cum), hop)
    starts = ends - frame_len + 1
    cum_prev = np.where(starts > 0, cum[starts - 1], 0.0)
    frame_energy = (cum[ends] - cum_prev) / frame_len
    return 10.0 * np.log10(frame_energy + 1e-20)


def zcr_frames(signal: NDArray, frame_len: int, hop: int) -> NDArray:
    """Zero crossing rate per frame — zero-copy using stride tricks (chunked)."""
    from numpy.lib.stride_tricks import as_strided
    n_pad = frame_len - len(signal) % hop
    sig = np.pad(signal, (0, n_pad), mode="constant")
    n_frames = (len(sig) - frame_len) // hop + 1
    out = np.zeros(n_frames)
    BLOCK = 20000
    for start in range(0, n_frames, BLOCK):
        stop = min(start + BLOCK, n_frames)
        shape = (stop - start, frame_len)
        strides = (sig.strides[0] * hop, sig.strides[0])
        off = start * hop
        frames = as_strided(sig[off:], shape=shape, strides=strides)
        signs = np.sign(frames).astype(np.int8)
        crossings = np.abs(signs[:, 1:] - signs[:, :-1]).sum(axis=1)
        out[start:stop] = crossings / frame_len
    return out


def spectral_features(
    signal: NDArray, sr: int, n_fft: int, hop: int
) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
    """Compute spectral centroid, bandwidth, rolloff, and flux (chunked).

    Returns ``(centroid, bandwidth, rolloff, flux)`` each of shape
    ``(n_frames,)``.
    """
    from scipy.signal import stft as _stft
    from numpy.lib.stride_tricks import as_strided

    n_pad = n_fft - len(signal) % hop
    sig = np.pad(signal, (0, n_pad), mode="constant")
    n_chunks_max = 200  # ~2000 frames per chunk at 10ms hop
    exp_frames_per_chunk = n_fft // hop + 1

    centroids, bandwidths, rolloffs, fluxes = [], [], [], []

    def _stats(mag):
        freqs = np.linspace(0, sr / 2, mag.shape[0])[:, None]
        tot = (mag).sum(axis=0) + 1e-20
        cent = (freqs * mag).sum(axis=0) / tot
        bw = np.sqrt(((freqs - cent) ** 2 * mag).sum(axis=0) / tot)
        cum = np.cumsum(mag, axis=0)
        total = cum[-1:]
        ridx = np.argmax(cum >= 0.85 * total, axis=0)
        ro = freqs[ridx].squeeze()
        return cent.squeeze(), bw.squeeze(), ro

    for start in range(0, len(sig), n_chunks_max * hop):
        chunk = sig[start:start + n_chunks_max * hop]
        if len(chunk) < n_fft:
            break
        _, _, Zxx = _stft(
            chunk, fs=sr, nperseg=n_fft, noverlap=n_fft - hop,
            window="hann", return_onesided=True, padded=False,
        )
        mag = np.abs(Zxx) + 1e-20
        cent, bw, ro = _stats(mag)
        centroids.append(cent)
        bandwidths.append(bw)
        rolloffs.append(ro)
        nf = cent.shape[0]
        # spectral flux between consecutive frames within chunk
        mag_log = np.log(mag[:, 1:] + 1e-20) - np.log(mag[:, :-1] + 1e-20)
        fl = np.sqrt((mag_log ** 2).sum(axis=0))
        fluxes.append(fl)

    centroid = np.concatenate(centroids)
    bandwidth = np.concatenate(bandwidths)
    rolloff = np.concatenate(rolloffs)
    flux = np.concatenate(fluxes)
    # Align lengths (flux has one fewer frame per chunk; take per-chunk mean padding)
    n_frames = centroid.shape[0]
    if flux.shape[0] < n_frames:
        flux = np.concatenate([flux, [np.mean(flux[-10:])] * (n_frames - flux.shape[0])])
    flux = flux[:n_frames]
    return centroid, bandwidth, rolloff, flux


def f0_autocorr(
    signal: NDArray,
    sr: int,
    frame_len: int,
    hop: int,
    f0_min: float = 60.0,
    f0_max: float = 400.0,
    voiced_threshold_db: float = -40.0,
) -> NDArray:
    """Pitch (F0) in Hz via autocorrelation, vectorised over VoIP frames.

    Returns ``np.nan`` for unvoiced frames.
    """
    from numpy.lib.stride_tricks import as_strided
    from scipy.signal import decimate

    rms_db = rms_frames(signal, frame_len, hop)
    n_pad = frame_len - len(signal) % hop
    sig = np.pad(signal, (0, n_pad), mode="constant")
    n_frames = len(rms_db)
    shape = (n_frames, frame_len)
    strides = (sig.strides[0] * hop, sig.strides[0])
    frames = as_strided(sig, shape=shape, strides=strides)

    voiced_mask = rms_db > voiced_threshold_db
    f0 = np.full(n_frames, np.nan)

    if voiced_mask.sum() == 0:
        return f0

    # Decimate to 8 kHz for faster autocorrelation
    decim_factor = 6
    decimated = decimate(frames[voiced_mask], decim_factor, axis=1, zero_phase=True)
    decim_sr = sr // decim_factor
    decim_len = decimated.shape[1]

    min_lag = max(1, int(decim_sr / f0_max))
    max_lag = min(decim_len - 1, int(decim_sr / f0_min))

    # FFT-based autocorrelation for the whole block
    n_fft = 1 << (decim_len - 1).bit_length()
    spec = np.fft.rfft(decimated, n=n_fft, axis=1)
    acf = np.fft.irfft(spec * np.conj(spec), n=n_fft, axis=1)
    acf = acf[:, :max_lag + 1]
    norm = acf[:, :1] + 1e-20
    acf_norm = acf / norm

    lag_range = acf_norm[:, min_lag:max_lag + 1]
    if lag_range.shape[1] < 3:
        return f0

    peak_idx_local = np.argmax(lag_range, axis=1) + min_lag

    # Parabolic interpolation (vectorised)
    rows = np.arange(len(peak_idx_local))
    alpha = acf_norm[rows, np.maximum(peak_idx_local - 1, 0)]
    beta = acf_norm[rows, peak_idx_local]
    gamma = acf_norm[rows, np.minimum(peak_idx_local + 1, max_lag)]
    denom = 2 * (2 * beta - alpha - gamma)
    shift = (alpha - gamma) / np.where(np.abs(denom) > 1e-10, denom, 1.0)
    shift = np.where(np.abs(denom) > 1e-10, shift, 0.0)
    peak_adj = peak_idx_local + shift

    valid = (peak_idx_local > 0) & (beta > 0.0)
    f0_vals = np.where(valid, decim_sr / np.maximum(peak_adj, 1.0), np.nan)
    f0[voiced_mask] = f0_vals
    return f0


def energy_slope(
    rms_db: NDArray,
    window_frames: int = 30,
) -> NDArray:
    """Trailing short-term slope of RMS energy in dB/s (causal, linear fit).

    Slope at frame ``i`` uses only frames in ``[i - window + 1, i]``.
    """
    n = len(rms_db)
    slope = np.zeros(n)
    for i in range(n):
        start = max(0, i - window_frames + 1)
        w = rms_db[start:i + 1]
        m = len(w)
        if m < 5:
            slope[i] = 0.0
        else:
            t_w = np.arange(m) - (m - 1) / 2
            slope[i] = np.sum(t_w * w) / (np.sum(t_w ** 2) + 1e-20)
    return slope


# ── VAD ──────────────────────────────────────────────────────────────────────

def compute_vad(
    audio_a: NDArray,
    audio_b: NDArray,
    sr: int,
    cfg: Config,
    noise_floor_a_db: float = -70.0,
    noise_floor_b_db: float = -70.0,
) -> Tuple[NDArray, NDArray]:
    """Voice activity detection for both channels.

    Uses a robust energy threshold derived from the low decile of
    RMS energy + a fixed margin.

    Returns
    -------
    vad_a, vad_b : NDArray[bool], one boolean per 10ms frame.
    """
    frame_len = int(sr * cfg.audio.frame_len_ms / 1000)
    hop = int(sr * cfg.audio.hop_ms / 1000)

    rms_a = rms_frames(audio_a, frame_len, hop)
    rms_b = rms_frames(audio_b, frame_len, hop)

    # Robust noise floor from low 25th percentile
    noise_floor_est_a = np.percentile(rms_a, 25)
    noise_floor_est_b = np.percentile(rms_b, 25)

    threshold_a = max(noise_floor_a_db, noise_floor_est_a) + cfg.features.vad_margin_above_noise_db
    threshold_b = max(noise_floor_b_db, noise_floor_est_b) + cfg.features.vad_margin_above_noise_db

    raw_a = rms_a > threshold_a
    raw_b = rms_b > threshold_b

    # Hangover: require min_speech_ms of continuous speech + hangover_frames
    hangover = cfg.audio.hangover_frames
    min_speech_frames = int(cfg.audio.min_speech_ms / cfg.audio.hop_ms)

    def _apply_hangover(mask: NDArray) -> NDArray:
        # Merge short silences < hangover within speech
        result = mask.copy()
        in_speech = False
        count = 0
        for i in range(len(result)):
            if result[i]:
                in_speech = True
                count = 0
            else:
                if in_speech:
                    count += 1
                    if count <= hangover:
                        result[i] = True
                    else:
                        in_speech = False
                        count = 0
        # Remove short speech segments
        in_speech = False
        seg_start = 0
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

    vad_a = _apply_hangover(raw_a)
    vad_b = _apply_hangover(raw_b)
    return vad_a, vad_b


def get_segments(
    vad: NDArray,
    hop_ms: float = 10.0,
) -> list:
    """Convert a boolean VAD vector into a list of (start_s, end_s) tuples."""
    segments = []
    in_seg = False
    start = 0
    for i, v in enumerate(vad):
        if v and not in_seg:
            start = i
            in_seg = True
        elif not v and in_seg:
            segments.append((start * hop_ms / 1000.0, i * hop_ms / 1000.0))
            in_seg = False
    if in_seg:
        segments.append((start * hop_ms / 1000.0, len(vad) * hop_ms / 1000.0))
    return segments


def compute_all_frame_features(
    audio: NDArray,
    sr: int,
    cfg: Config,
) -> dict:
    """Compute all frame-level features for one channel.

    Returns a dict with arrays of shape ``(n_frames,)``:
    - ``time_s``: centre time of each frame
    - ``rms_db``, ``zcr``, ``centroid``, ``bandwidth``, ``rolloff``,
      ``spectral_flux``, ``f0``, ``energy_slope``
    """
    frame_len = int(sr * cfg.audio.frame_len_ms / 1000)
    hop = int(sr * cfg.audio.hop_ms / 1000)

    n_frames = (len(audio) - frame_len) // hop + 1
    time_s = (np.arange(n_frames) * hop + frame_len // 2) / sr

    rms_db = rms_frames(audio, frame_len, hop)[:n_frames]
    zcr_val = zcr_frames(audio, frame_len, hop)[:n_frames]
    centroid, bandwidth, rolloff, flux = spectral_features(
        audio, sr, cfg.features.spectral_n_fft, hop
    )
    centroid = centroid[:n_frames]
    bandwidth = bandwidth[:n_frames]
    rolloff = rolloff[:n_frames]
    flux = flux[:n_frames]
    f0 = f0_autocorr(
        audio, sr, frame_len, hop,
        f0_min=cfg.features.f0_range[0],
        f0_max=cfg.features.f0_range[1],
    )[:n_frames]
    eslope = energy_slope(rms_db)[:n_frames]

    return {
        "time_s": time_s,
        "rms_db": rms_db,
        "zcr": zcr_val,
        "centroid": centroid,
        "bandwidth": bandwidth,
        "rolloff": rolloff,
        "spectral_flux": flux,
        "f0": f0,
        "energy_slope": eslope,
    }


def compute_all_frame_features_wrapper(
    audio_a: NDArray,
    audio_b: NDArray,
    sr: int,
    cfg: Config,
    noise_floor_a_db: float = -70.0,
    noise_floor_b_db: float = -70.0,
) -> dict:
    """Compute features + VAD for both channels."""
    vad_a, vad_b = compute_vad(
        audio_a, audio_b, sr, cfg,
        noise_floor_a_db=noise_floor_a_db,
        noise_floor_b_db=noise_floor_b_db,
    )
    feat_a = compute_all_frame_features(audio_a, sr, cfg)
    feat_b = compute_all_frame_features(audio_b, sr, cfg)
    return {"a": feat_a, "b": feat_b, "vad_a": vad_a, "vad_b": vad_b}
