# core/naming_utils.py

"""
Naming utilities for semantic and attribute clusters.

These helpers are intentionally generic and no longer tied to any
specific layer names (no "group", "parent", or "subcluster" terminology).

The primary public function is:

    compute_auto_name_for_category_list(categories: List[str])
        -> (top_word: str, top_words: List[str], auto_label: str)

It takes a list of category-like strings (e.g., product category names)
and returns:
    - top_word:   the single most frequent informative token
    - top_words:  all equally most-frequent tokens (after filtering)
    - auto_label: a human-readable label composed from top_words
"""

from __future__ import annotations

from collections import Counter
from typing import List, Tuple

import regex as re

from .text_utils import lemmatizer


# ------------------------------------------------------------
# Tokenization and filtering
# ------------------------------------------------------------

_STOPWORDS = {
    "and",
    "or",
    "the",
    "a",
    "an",
    "for",
    "with",
    "to",
    "of",
    "in",
    "on",
    "by",
    "from",
    "at",
    "mm",    # often unit-like junk tokens in names
    "cm",
    "inch",
    "inches",
}


def _tokenize_for_cluster(text: str) -> List[str]:
    """
    Simple tokenizer for computing frequent words in cluster member names.

    Steps:
    - Lowercase
    - Extract word-like tokens with a regex
    - Lemmatize each token
    - Filter out stopwords and very short tokens
    """
    if not text:
        return []

    words = re.findall(r"\w+", str(text).lower())
    tokens: List[str] = []

    for w in words:
        lemma = lemmatizer.lemmatize(w)
        if lemma in _STOPWORDS:
            continue
        if len(lemma) <= 1:
            continue
        tokens.append(lemma)

    return tokens


# ------------------------------------------------------------
# Public: auto-naming for a list of category-like labels
# ------------------------------------------------------------

def compute_auto_name_for_category_list(
    categories: List[str],
) -> Tuple[str, List[str], str]:
    """
    Compute a simple, frequency-based label for a list of category strings.

    This was historically used to auto-name clusters of categories and
    remains useful for quick, heuristic naming in the new architecture.

    Parameters
    ----------
    categories :
        List of strings (e.g., category or label names) that belong to
        the same cluster.

    Returns
    -------
    top_word :
        The single most frequent token among all non-stopword tokens.
    top_words :
        A list of all equally most-frequent tokens.
    auto_label :
        A human-readable label constructed by joining top_words with
        commas. The label is title-cased for display.

    Notes
    -----
    - If no informative tokens are found, we fall back to "misc".
    - This is a very lightweight heuristic compared to TF-IDF or
      embedding-based naming (see text_utils.tfidf_cluster_label).
    """
    word_counts: Counter = Counter()

    for cat in categories:
        tokens = _tokenize_for_cluster(cat)
        word_counts.update(tokens)

    if not word_counts:
        # Degenerate case: nothing to work with
        print_top_words = ["misc"]
        top_word = "misc"
    else:
        max_freq = max(word_counts.values())
        print_top_words = [
            word
            for word, freq in word_counts.items()
            if freq == max_freq
        ]
        # Stable ordering for determinism
        print_top_words = sorted(print_top_words)
        top_word = print_top_words[0]

    # Construct a readable label
    auto_label = ", ".join(print_top_words).title()

    return top_word, print_top_words, auto_label
