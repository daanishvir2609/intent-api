"""Data loading and the scikit-learn pipeline: raw text in, intent out."""
import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

OUT_OF_SCOPE = "out_of_scope"


def load(path: str, split: str) -> tuple[list[str], list[str]]:
    """CLINC150 splits: train, val, test (150 intents) and oos_train, oos_val, oos_test."""
    with open(path) as f:
        rows = json.load(f)[split]
    texts = [text for text, _ in rows]
    labels = [label if not split.startswith("oos") else OUT_OF_SCOPE for _, label in rows]
    return texts, labels


def build_pipeline(C: float = 10.0) -> Pipeline:
    return Pipeline([
        # step 1: turn each request into numbers (word and word-pair weights)
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2)),
        # step 2: one weight per (feature, intent); softmax gives a probability per intent
        ("clf", LogisticRegression(C=C, max_iter=2000)),
    ])


def predict_with_fallback(model: Pipeline, texts, threshold: float) -> list[str]:
    """If the model isn't confident, say "out of scope" instead of guessing an action."""
    proba = model.predict_proba(texts)
    best = proba.argmax(axis=1)
    return [model.classes_[i] if p[i] >= threshold else OUT_OF_SCOPE for i, p in zip(best, proba)]