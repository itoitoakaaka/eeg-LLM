# eeg-LLM

EEG feature extraction + retrieval-augmented LLM reasoning.

## What this repository does

This project turns EEG-derived measurements into a structured representation, retrieves relevant methodological context, and then asks a language model to produce a grounded explanation.

The LLM is not asked to infer directly from raw EEG. Instead, the pipeline keeps three layers explicit:

1. **Measurement** — numerical EEG features
2. **Evidence context** — retrieved methodological notes
3. **LLM reasoning** — a structured explanation constrained by the first two layers

## Pipeline

    synthetic/raw EEG
          |
          v
    EEG feature extraction
          |
          v
    structured condition comparison
          |
          +----> knowledge retrieval
          |          |
          v          v
       grounded LLM prompt
          |
          v
    Observation / Interpretation / Cannot conclude

## Files

- `eeg_features.py` — spectral feature extraction and condition comparison
- `knowledge_rag.py` — TF-IDF retrieval over a small local knowledge base
- `llm_reasoner.py` — grounded prompt construction and optional local LLM inference
- `knowledge_base.json` — methodological context used by retrieval
- `demo.py` — end-to-end synthetic example
- `tests/test_pipeline.py` — tests that do not require downloading an LLM

## Core design rule

The model must distinguish:

### Observation
What changed numerically in the EEG-derived features?

### Interpretation
What explanations are consistent with those measurements and the retrieved context?

### Cannot conclude
Which stronger claims are not established by the data?

The prompt explicitly blocks unsupported diagnoses, emotion inference, and ungrounded cognitive-state claims.

## Setup

Core pipeline:

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

Optional local LLM support:

    pip install -r requirements-llm.txt

## Run without downloading an LLM

    python demo.py

This executes EEG feature extraction, retrieval, and prompt construction.

## Run with a local Hugging Face language model

    python demo.py --model <hugging-face-model-name>

The model name is intentionally user-supplied. This repository does not depend on one particular model family.

## Why this is called eeg-LLM

The language model is now an actual reasoning layer in the architecture rather than a decorative text formatter. At the same time, EEG preprocessing and numerical measurements remain inspectable and separable from generated interpretation.

## Scope

The demo uses synthetic EEG-like signals and a small methodological knowledge base. It is a portfolio prototype, not a medical or diagnostic system.
