# core/text_utils.py

"""
Text utilities for semantic and attribute clustering.

This module provides:

    - A shared SentenceTransformer model for embeddings
    - A shared WordNet lemmatizer
    - Helpers for normalizing and tokenizing text
    - TF-IDF–based cluster labeling that is robust to sparse input
"""

# Type hints
from __future__ import annotations
from typing import List, Sequence

# External dependencies
import numpy as np
import regex as re
from collections import Counter

# Sklearn dependencies
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer


# ============================================================
#   Shared models / global instances
# ============================================================

_sentence_model: SentenceTransformer | None = None

# lemmatizer = WordNetLemmatizer()


def _get_sentence_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """
    Lazily load and cache the SentenceTransformer model used for
    semantic embedding of labels (categories, cluster names, etc.).
    """
    global _sentence_model
    if _sentence_model is None:
        _sentence_model = SentenceTransformer(model_name)
    return _sentence_model


# ============================================================
#   Basic text normalization / tokenization helpers
# ============================================================

def normalize_text(text: str) -> str:
    """
    Normalize whitespace and coerce to string.

    This is intentionally lightweight: we primarily use it to make
    sure we don't feed empty strings or pathological whitespace into
    TF-IDF or embedding models.
    """
    s = str(text)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def tokenize(text: str) -> List[str]:
    """
    Very simple tokenizer for fallback frequency calculations.

    - Lowercase
    - Extract sequences of letters/digits as tokens
    """
    s = normalize_text(text).lower()
    return re.findall(r"\p{L}+", s)


# ============================================================
#   Embeddings for labels (semantic clustering)
# ============================================================

def build_embeddings_for_labels(labels: Sequence[str]) -> np.ndarray:
    """
    Build sentence-level embeddings for a sequence of label strings,
    using a shared SentenceTransformer model.

    Parameters
    ----------
    labels :
        Any iterable of strings (e.g., category names, cluster labels).

    Returns
    -------
    np.ndarray
        2D array of shape (len(labels), embedding_dim).
        Returns an empty array if `labels` is empty.
    """
    labels = list(labels)
    if not labels:
        return np.zeros((0, 0), dtype=float)

    model = _get_sentence_model()
    cleaned = [normalize_text(x) for x in labels]
    emb = model.encode(cleaned, show_progress_bar=False)
    return np.asarray(emb, dtype=float)


# ============================================================
#   TF-IDF–based cluster labeling (robust)
# ============================================================

def tfidf_cluster_label(
    texts: Sequence[str],
    max_words: int = 4,
    min_df: int = 1,
    max_df: float = 0.9,
) -> str:
    """
    Compute a short, human-readable label from a collection of texts
    using TF-IDF scores.

    This is used to name semantic or attribute clusters in a way that
    reflects the most informative terms appearing in cluster members.

    The function is designed to be robust:
      - If input is empty → returns "misc"
      - If TF-IDF pruning removes all terms → relaxes pruning and retries
      - If that still fails → falls back to simple token frequency

    Parameters
    ----------
    texts :
        Sequence of strings belonging to a single cluster.
    max_words :
        Maximum number of words to include in the label.
    min_df :
        Minimum document frequency for TF-IDF features.
    max_df :
        Maximum document frequency (as a proportion) for TF-IDF features.

    Returns
    -------
    str
        A title-cased cluster label, or "misc" if no good label
        can be determined.
    """
    # Normalize and filter empties
    docs = [normalize_text(t) for t in texts if normalize_text(t)]
    if not docs:
        return "misc"

    # First pass: user-specified min_df / max_df
    vec = TfidfVectorizer(
        lowercase=True,
        token_pattern=r"\b\w+\b",
        min_df=min_df,
        max_df=max_df,
    )

    try:
        X = vec.fit_transform(docs)
    except ValueError:
        # Typical case: "After pruning, no terms remain".
        # Relax pruning and retry.
        vec = TfidfVectorizer(
            lowercase=True,
            token_pattern=r"\b\w+\b",
            min_df=1,
            max_df=1.0,
        )
        try:
            X = vec.fit_transform(docs)
        except Exception:
            # Final fallback: simple frequency over tokens
            tokens: List[str] = []
            for d in docs:
                tokens.extend(tokenize(d))
            if not tokens:
                return "misc"
            counts = Counter(tokens)
            top_tokens = [w for w, _ in counts.most_common(max_words)]
            return " ".join(top_tokens).title()

    if X.shape[1] == 0:
        # No features survived
        return "misc"

    terms = np.array(vec.get_feature_names_out())
    scores = np.asarray(X.mean(axis=0)).ravel()
    if scores.size == 0:
        return "misc"

    # Rank terms by average TF-IDF score
    order = scores.argsort()[::-1]

    label_tokens: List[str] = []
    for idx in order:
        if scores[idx] <= 0:
            continue
        label_tokens.append(terms[idx])
        if len(label_tokens) >= max_words:
            break

    if not label_tokens:
        return "misc"

    # Title-case to look nicer as a cluster name
    return " ".join(label_tokens).title()

