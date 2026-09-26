from __future__ import annotations

import numpy as np
from scipy.signal import welch


BANDS = {
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
}


def bandpower(signal, sfreq, fmin, fmax):
    """Integrate Welch PSD within a frequency band."""
    signal = np.asarray(signal, dtype=float)
    nperseg = min(signal.size, max(32, int(sfreq * 2)))
    freqs, psd = welch(signal, fs=sfreq, nperseg=nperseg)
    mask = (freqs >= fmin) & (freqs < fmax)
    if not mask.any():
        return float("nan")
    return float(np.trapezoid(psd[mask], freqs[mask]))


def summarize_eeg(signal, sfreq=250.0):
    """Return transparent numerical EEG summary features."""
    signal = np.asarray(signal, dtype=float)
    absolute = {
        name: bandpower(signal, sfreq, low, high)
        for name, (low, high) in BANDS.items()
    }

    finite_total = sum(v for v in absolute.values() if np.isfinite(v))
    relative = {
        name: float(value / finite_total) if finite_total > 0 else float("nan")
        for name, value in absolute.items()
    }

    return {
        "n_samples": int(signal.size),
        "sampling_rate_hz": float(sfreq),
        "mean_uv": float(signal.mean()),
        "std_uv": float(signal.std()),
        "peak_to_peak_uv": float(np.ptp(signal)),
        "bandpower": absolute,
        "relative_bandpower": relative,
    }


def compare_conditions(summary_a, summary_b, label_a="A", label_b="B"):
    """Build an auditable condition-comparison record."""
    comparison = {
        "condition_a": label_a,
        "condition_b": label_b,
        "features": {},
    }

    for band in BANDS:
        a = float(summary_a["relative_bandpower"][band])
        b = float(summary_b["relative_bandpower"][band])
        comparison["features"][f"relative_{band}_power"] = {
            label_a: a,
            label_b: b,
            "difference_b_minus_a": b - a,
        }

    for metric in ("std_uv", "peak_to_peak_uv"):
        a = float(summary_a[metric])
        b = float(summary_b[metric])
        comparison["features"][metric] = {
            label_a: a,
            label_b: b,
            "difference_b_minus_a": b - a,
        }

    return comparison
