from __future__ import annotations

import json

import numpy as np
from scipy.signal import welch


BANDS = {
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
}


def bandpower(signal, sfreq, fmin, fmax):
    freqs, psd = welch(signal, fs=sfreq, nperseg=min(len(signal), int(sfreq * 2)))
    mask = (freqs >= fmin) & (freqs < fmax)
    if not mask.any():
        return float("nan")
    return float(np.trapz(psd[mask], freqs[mask]))


def summarize_eeg(signal, sfreq=250.0):
    signal = np.asarray(signal, dtype=float)
    result = {
        "n_samples": int(signal.size),
        "sampling_rate_hz": float(sfreq),
        "mean_uv": float(signal.mean()),
        "std_uv": float(signal.std()),
        "peak_to_peak_uv": float(np.ptp(signal)),
        "bandpower": {},
    }
    for name, (low, high) in BANDS.items():
        result["bandpower"][name] = bandpower(signal, sfreq, low, high)
    total = sum(v for v in result["bandpower"].values() if np.isfinite(v))
    if total > 0:
        result["relative_bandpower"] = {
            key: float(value / total) for key, value in result["bandpower"].items()
        }
    return result


def to_json(summary):
    return json.dumps(summary, ensure_ascii=False, indent=2)
