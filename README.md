# eeg-LLM

A small bridge project between EEG signal features and language-based reporting.

## Purpose

This repository is not an LLM-training project. It demonstrates how structured EEG features can be converted into a compact, auditable text representation that could later be consumed by an LLM or other reasoning system.

The emphasis is on keeping the signal-processing stage explicit and separating measured features from generated interpretation.

## Workflow

1. Generate or load EEG-like time-series data.
2. Extract transparent summary features.
3. Convert those features into a structured JSON record.
4. Build a text prompt/report template from the JSON record.
5. Optionally connect the structured output to an external LLM in a separate application layer.

## Why this matters

For human-centered AI and physical-AI applications, physiological signals should not be passed to a language model as an opaque claim. The measurable signal features and the generated interpretation should remain separable.

## Quick start

    python demo.py

The current demo uses synthetic EEG-like data only.
