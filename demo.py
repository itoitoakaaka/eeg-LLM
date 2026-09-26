from __future__ import annotations

import argparse
import json

import numpy as np

from eeg_features import compare_conditions, summarize_eeg
from knowledge_rag import load_knowledge_base, retrieve_context
from llm_reasoner import LocalTransformersReasoner, build_grounded_prompt


def synthetic_eeg(
    sfreq=250.0,
    duration_s=12.0,
    alpha_amp=4.0,
    beta_amp=1.5,
    random_state=42,
):
    rng = np.random.default_rng(random_state)
    t = np.arange(0, duration_s, 1.0 / sfreq)
    return (
        alpha_amp * np.sin(2 * np.pi * 10.0 * t)
        + beta_amp * np.sin(2 * np.pi * 20.0 * t)
        + rng.normal(0.0, 2.0, size=len(t))
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=None,
        help="Optional Hugging Face text-generation model name.",
    )
    args = parser.parse_args()

    signal_a = synthetic_eeg(
        alpha_amp=4.5,
        beta_amp=1.2,
        random_state=10,
    )
    signal_b = synthetic_eeg(
        alpha_amp=2.8,
        beta_amp=2.3,
        random_state=11,
    )

    summary_a = summarize_eeg(signal_a)
    summary_b = summarize_eeg(signal_b)

    comparison = compare_conditions(
        summary_a,
        summary_b,
        label_a="Condition A",
        label_b="Condition B",
    )

    question = (
        "How do the EEG-derived features differ between Condition A and "
        "Condition B, and what can or cannot be inferred from that difference?"
    )

    knowledge = load_knowledge_base()
    retrieved = retrieve_context(question, knowledge, top_k=3)
    prompt = build_grounded_prompt(comparison, retrieved, question)

    print("=== Structured EEG comparison ===")
    print(json.dumps(comparison, indent=2))

    print("\n=== Retrieved context ===")
    for item in retrieved:
        print(f"[{item['id']}] score={item['score']:.3f}")
        print(item["text"])

    if args.model is None:
        print("\n=== Grounded LLM prompt ===")
        print(prompt)
        print(
            "\nNo model was requested. Re-run with "
            "--model <hugging-face-model-name> to invoke a local LLM."
        )
        return

    print("\n=== LLM reasoning ===")
    reasoner = LocalTransformersReasoner(args.model)
    print(reasoner.generate(prompt))


if __name__ == "__main__":
    main()
