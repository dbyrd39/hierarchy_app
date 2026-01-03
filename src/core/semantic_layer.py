# core/semantic_layer.py

"""
This file implements the Semantic Layer.

The Semantic Layer groups unique string labels using embedding-based
clustering to identify semantically similar categories, then maps those
cluster assignments back onto the original dataframe. Each semantic
cluster is also given a human-readable name derived from its member
labels using TF-IDF keyword scoring.

This layer is typically applied to categorical or hierarchical label
columns (e.g., category names) and is intended to sit above or alongside
attribute-based clustering layers.

Public API used by the hierarchy engine:

    build_semantic_layer(
        df,
        *,
        input_label_col,
        n_clusters,
        output_prefix,
        random_state=42
    ) -> pd.DataFrame
        Returns a copy of df with two new columns:
        `{output_prefix}_id` containing integer semantic cluster IDs
        and `{output_prefix}_name` containing human-readable cluster names.

Internal helpers:

    _choose_k(n_items, k_min=2, k_max=12) -> int
        Heuristic for selecting a reasonable number of clusters when
        an explicit value is not provided.
"""

# Type hints
from __future__ import annotations

# External dependencies
import numpy as np
import pandas as pd

# Sklearn dependencies
from sklearn.cluster import KMeans

# Internal dependencies
from .text_utils import build_embeddings_for_labels, tfidf_cluster_label

# ============================================================
# K selection helper
# ============================================================

def _choose_k(
    n_items: int,
    k_min: int = 2,
    k_max: int = 12,
) -> int:
    """
    Select a reasonable number of clusters using a simple heuristic.

    This helper is used when the caller does not provide an explicit cluster
    count. The default heuristic uses `sqrt(n_items)` and clamps the result
    to a configurable [k_min, k_max] range, while also ensuring we never ask
    for more clusters than items.

    Parameters
    ----------
    n_items :
        Number of items to be clustered.
    k_min :
        Minimum allowable number of clusters (when `n_items > 1`).
    k_max :
        Maximum allowable number of clusters.

    Returns
    -------
    int :
        The chosen number of clusters.
    """
    if n_items <= 1:
        return 1

    k = int(np.sqrt(n_items))
    k = max(k_min, min(k_max, k))
    k = min(k, n_items)
    return k


# ============================================================
# Public: Semantic layer construction
# ============================================================

def build_semantic_layer(
    df: pd.DataFrame,
    *,
    input_label_col: str,
    n_clusters: int | None,
    output_prefix: str,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Construct a semantic clustering layer from a string label column.

    This function groups unique label values using embedding-based clustering
    (KMeans over vector embeddings), then maps the resulting cluster IDs back
    onto the original DataFrame. It also generates a human-readable cluster
    name per group using a TF-IDF-based labeler.

    Two output columns are created:

    - `{output_prefix}_id` :
        Integer cluster ID per row. IDs are normalized to 1..K, with `0`
        reserved for unmapped/empty labels.
    - `{output_prefix}_name` :
        Human-readable cluster name per row derived from cluster members.

    Parameters
    ----------
    df :
        Primary DataFrame containing the label column to be clustered.
    input_label_col :
        Column name containing raw string labels (e.g., category names).
    n_clusters :
        Desired number of clusters. If None or < 1, a heuristic is used.
        The final value is always clamped to [1, n_unique_labels].
    output_prefix :
        Prefix used to name the output columns.
    random_state :
        Random seed used for KMeans to ensure reproducible clustering.

    Returns
    -------
    pd.DataFrame :
        A copy of `df` with `{output_prefix}_id` and `{output_prefix}_name`
        appended.
    """
    df = df.copy()

    labels = (
        df[input_label_col]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    unique_labels = sorted({v for v in labels if v})
    if not unique_labels:
        df[f"{output_prefix}_id"] = 0
        df[f"{output_prefix}_name"] = "All"
        return df

    # Choose k
    if n_clusters is None or n_clusters < 1:
        k = _choose_k(len(unique_labels))
    else:
        k = min(max(1, n_clusters), len(unique_labels))

    # Build embeddings and cluster
    emb = build_embeddings_for_labels(unique_labels)

    if k <= 1:
        cluster_raw = np.zeros(len(unique_labels), dtype=int)
    else:
        km = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
        cluster_raw = km.fit_predict(emb)

    # Normalize cluster IDs to 1..K (reserve 0 for missing/unmapped)
    uniq_raw = sorted(set(int(x) for x in cluster_raw))
    raw2id = {raw: i + 1 for i, raw in enumerate(uniq_raw)}

    label_to_cid = {
        lbl: raw2id[int(raw)]
        for lbl, raw in zip(unique_labels, cluster_raw)
    }

    df[f"{output_prefix}_id"] = labels.map(label_to_cid).fillna(0).astype(int)

    # Generate names per cluster from member labels
    clusters: dict[int, list[str]] = {}
    for lbl, cid in label_to_cid.items():
        clusters.setdefault(cid, []).append(lbl)

    name_map: dict[int, str] = {}
    for cid, members in clusters.items():
        try:
            name_map[cid] = tfidf_cluster_label(members)
        except Exception:
            name_map[cid] = " / ".join(members[:3])

    df[f"{output_prefix}_name"] = df[f"{output_prefix}_id"].map(name_map).astype(str)
    return df


# ============================================================
# Semantic relabel
# ============================================================

# def semantic_relabel(df: pd.DataFrame, id_col: str, name_col: str | None = None, n_clusters=None):

#     if name_col and name_col in df.columns:
#         labels = df[name_col].fillna("").astype(str)
#     else:
#         labels = df[id_col].astype(str)

#     unique = sorted({x for x in labels if x})
#     if not unique:
#         df2 = df.copy()
#         df2[f"{id_col}_semantic"] = 0
#         df2[f"{id_col}_semantic_name"] = "All"
#         return df2

#     if n_clusters is None or n_clusters < 1:
#         k = _choose_k(len(unique))
#     else:
#         k = min(max(1, n_clusters), len(unique))

#     emb = build_embeddings_for_labels(unique)
#     if k <= 1:
#         raw = np.zeros(len(unique), dtype=int)
#     else:
#         km = KMeans(n_clusters=k, random_state=42, n_init="auto")
#         raw = km.fit_predict(emb)

#     uniq_raw = sorted(set(int(x) for x in raw))
#     raw2id = {old: i + 1 for i, old in enumerate(uniq_raw)}

#     label2cid = {lbl: raw2id[int(r)] for lbl, r in zip(unique, raw)}

#     df2 = df.copy()
#     df2[f"{id_col}_semantic"] = labels.map(label2cid).fillna(0).astype(int)

#     clusters = {}
#     for lbl, cid in label2cid.items():
#         clusters.setdefault(cid, []).append(lbl)

#     names = {}
#     for cid, mem in clusters.items():
#         try:
#             names[cid] = tfidf_cluster_label(mem)
#         except:
#             names[cid] = " / ".join(mem[:3])

#     df2[f"{id_col}_semantic_name"] = df2[f"{id_col}_semantic"].map(names).astype(str)
#     return df2


# ============================================================
# Split helper (embeddings)
# ============================================================

# def split_cluster_embeddings(labels: list[str], k: int):

#     emb = build_embeddings_for_labels(labels)

#     km = KMeans(n_clusters=k, random_state=42, n_init="auto")
#     raw = km.fit_predict(emb)

#     # The return is simply an array of group indices (0..k-1)
#     return raw
