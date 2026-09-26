from __future__ import annotations

import json


def build_prompt(summary):
    payload = json.dumps(summary, ensure_ascii=False, indent=2)
    return f"""You are given structured EEG summary features.

Do not infer diagnoses, emotions, or cognitive states that are not supported by the measurements.
Describe only the numerical signal properties and identify which conclusions would require additional evidence.

EEG summary:
{payload}
"""


def build_plain_report(summary):
    rel = summary.get("relative_bandpower", {})
    lines = [
        f"Samples: {summary['n_samples']}",
        f"Sampling rate: {summary['sampling_rate_hz']:.1f} Hz",
        f"Signal SD: {summary['std_uv']:.3f} uV",
        f"Peak-to-peak: {summary['peak_to_peak_uv']:.3f} uV",
    ]
    if rel:
        lines.append(
            "Relative band power: "
            + ", ".join(f"{name}={value:.3f}" for name, value in rel.items())
        )
    return "\n".join(lines)
