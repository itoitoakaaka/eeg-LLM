from __future__ import annotations

import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_knowledge_base(path="knowledge_base.json"):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8"))


def retrieve_context(query, entries, top_k=3):
    """Retrieve the most relevant local methodological notes."""
    if not entries:
        return []

    corpus = [entry["text"] for entry in entries]
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(corpus + [query])
    scores = cosine_similarity(matrix[-1], matrix[:-1]).ravel()

    order = scores.argsort()[::-1][:top_k]
    return [
        {
            "id": entries[i]["id"],
            "text": entries[i]["text"],
            "score": float(scores[i]),
        }
        for i in order
    ]
