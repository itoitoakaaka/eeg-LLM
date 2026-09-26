import numpy as np

from features import summarize_eeg, to_json
from prompt_builder import build_plain_report, build_prompt


def synthetic_eeg(sfreq=250.0, duration_s=10.0, random_state=42):
    rng = np.random.default_rng(random_state)
    t = np.arange(0, duration_s, 1.0 / sfreq)
    signal = (
        4.0 * np.sin(2 * np.pi * 10.0 * t)
        + 1.5 * np.sin(2 * np.pi * 20.0 * t)
        + rng.normal(0.0, 2.0, size=len(t))
    )
    return signal


def main():
    signal = synthetic_eeg()
    summary = summarize_eeg(signal)

    print("=== Structured features ===")
    print(to_json(summary))
    print("\n=== Plain report ===")
    print(build_plain_report(summary))
    print("\n=== Prompt for an optional language-model layer ===")
    print(build_prompt(summary))


if __name__ == "__main__":
    main()
