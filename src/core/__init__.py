"""
Core processing modules for the Product Hierarchy Builder (BL3).

This package contains:

    - hierarchy_engine      → Unified backend engine
    - semantic_layer        → Generic semantic clustering utilities
    - attribute_layer       → Attribute sparsity clustering utilities
    - category_layer        → Category normalization + summaries
    - naming_utils          → Generic cluster naming helpers
    - text_utils            → Embeddings + TF-IDF + text normalization
"""

from .hierarchy_engine import HierarchyEngine
from .semantic_layer import (
    build_semantic_layer
)
from .category_layer import (
    ensure_category_name_column
)
from .text_utils import (
    normalize_text,
    tokenize,
    build_embeddings_for_labels,
    tfidf_cluster_label,
)

