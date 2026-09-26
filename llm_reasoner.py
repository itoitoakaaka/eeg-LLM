from __future__ import annotations

import json


def build_grounded_prompt(comparison, retrieved_context, question):
    """Build a constrained prompt for EEG-related LLM reasoning."""
    observation_json = json.dumps(comparison, ensure_ascii=False, indent=2)
    evidence = "\n\n".join(
        f"[{item['id']}] {item['text']}" for item in retrieved_context
    )

    return f"""You are analyzing structured EEG-derived measurements.

Question:
{question}

Measured condition comparison:
{observation_json}

Retrieved methodological context:
{evidence}

Instructions:
- Keep measured observations separate from interpretation.
- Use only the measurements and retrieved context above.
- Do not infer diagnoses, emotions, attention states, inhibition, excitation, or mechanisms unless the supplied evidence supports that claim.
- Do not turn association into causation.
- Explicitly state uncertainty.
- If the evidence is insufficient, say so.
- Return exactly these three sections:

Observation:
Describe the numerical differences only.

Interpretation:
Describe cautious explanations that are consistent with the measurements and retrieved context.

Cannot conclude:
List stronger claims that are not established by the available evidence.
"""


class LocalTransformersReasoner:
    """Optional local Hugging Face text-generation backend."""

    def __init__(self, model_name):
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise RuntimeError(
                "Install optional dependencies with: "
                "pip install -r requirements-llm.txt"
            ) from exc

        self.generator = pipeline(
            "text-generation",
            model=model_name,
            tokenizer=model_name,
        )

    def generate(self, prompt, max_new_tokens=350):
        result = self.generator(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_full_text=False,
        )
        return result[0]["generated_text"]
