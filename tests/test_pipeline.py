import unittest

import numpy as np

from eeg_features import compare_conditions, summarize_eeg
from knowledge_rag import load_knowledge_base, retrieve_context
from llm_reasoner import build_grounded_prompt


class EEGLLMTests(unittest.TestCase):
    def test_alpha_difference_is_detected(self):
        sfreq = 250.0
        t = np.arange(0, 10.0, 1.0 / sfreq)
        a = 5.0 * np.sin(2 * np.pi * 10.0 * t)
        b = 2.0 * np.sin(2 * np.pi * 10.0 * t)

        summary_a = summarize_eeg(a, sfreq)
        summary_b = summarize_eeg(b, sfreq)
        comparison = compare_conditions(summary_a, summary_b)

        delta = comparison["features"]["relative_alpha_power"]["difference_b_minus_a"]
        self.assertLessEqual(delta, 0.0)

    def test_retrieval_returns_context(self):
        entries = load_knowledge_base()
        retrieved = retrieve_context(
            "Can alpha power prove a cognitive state?",
            entries,
            top_k=2,
        )
        self.assertEqual(len(retrieved), 2)
        self.assertTrue(all("id" in item for item in retrieved))

    def test_prompt_separates_inference_levels(self):
        comparison = {"condition_a": "A", "condition_b": "B", "features": {}}
        context = [{"id": "x", "text": "Do not over-interpret EEG.", "score": 1.0}]
        prompt = build_grounded_prompt(
            comparison,
            context,
            "What changed?",
        )

        self.assertIn("Observation:", prompt)
        self.assertIn("Interpretation:", prompt)
        self.assertIn("Cannot conclude:", prompt)


if __name__ == "__main__":
    unittest.main()
