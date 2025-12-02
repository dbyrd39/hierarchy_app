# core/semantic_layer.py

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from .text_utils import build_embeddings_for_labels, tfidf_cluster_label


def _choose_k(n_items: int, k_min: int = 2, k_max: int = 12):
    if n_items <= 1:
        return 1
    k = int(np.sqrt(n_items))
    k = max(k_min, min(k_max, k))
    k = min(k, n_items)
    return k


# ============================================================
# Semantic layer construction
# ============================================================

def build_semantic_layer(
    df: pd.DataFrame,
    *,
    input_label_col: str,
    n_clusters: int | None,
    output_prefix: str,
    random_state: int = 42,
) -> pd.DataFrame:

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

    # k
    if n_clusters is None or n_clusters < 1:
        k = _choose_k(len(unique_labels))
    else:
        k = min(max(1, n_clusters), len(unique_labels))

    # embeddings
    emb = build_embeddings_for_labels(unique_labels)

    if k <= 1:
        cluster_raw = np.zeros(len(unique_labels), dtype=int)
    else:
        km = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
        cluster_raw = km.fit_predict(emb)

    # normalize IDs (1..K)
    uniq_raw = sorted(set(int(x) for x in cluster_raw))
    raw2id = {raw: i + 1 for i, raw in enumerate(uniq_raw)}

    label_to_cid = {
        lbl: raw2id[int(raw)]
        for lbl, raw in zip(unique_labels, cluster_raw)
    }

    df[f"{output_prefix}_id"] = labels.map(label_to_cid).fillna(0).astype(int)

    # names
    clusters = {}
    for lbl, cid in label_to_cid.items():
        clusters.setdefault(cid, []).append(lbl)

    name_map = {}
    for cid, members in clusters.items():
        try:
            name_map[cid] = tfidf_cluster_label(members)
        except:
            name_map[cid] = " / ".join(members[:3])

    df[f"{output_prefix}_name"] = df[f"{output_prefix}_id"].map(name_map).astype(str)
    return df


# ============================================================
# Semantic relabel
# ============================================================

def semantic_relabel(df: pd.DataFrame, id_col: str, name_col: str | None = None, n_clusters=None):

    if name_col and name_col in df.columns:
        labels = df[name_col].fillna("").astype(str)
    else:
        labels = df[id_col].astype(str)

    unique = sorted({x for x in labels if x})
    if not unique:
        df2 = df.copy()
        df2[f"{id_col}_semantic"] = 0
        df2[f"{id_col}_semantic_name"] = "All"
        return df2

    if n_clusters is None or n_clusters < 1:
        k = _choose_k(len(unique))
    else:
        k = min(max(1, n_clusters), len(unique))

    emb = build_embeddings_for_labels(unique)
    if k <= 1:
        raw = np.zeros(len(unique), dtype=int)
    else:
        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        raw = km.fit_predict(emb)

    uniq_raw = sorted(set(int(x) for x in raw))
    raw2id = {old: i + 1 for i, old in enumerate(uniq_raw)}

    label2cid = {lbl: raw2id[int(r)] for lbl, r in zip(unique, raw)}

    df2 = df.copy()
    df2[f"{id_col}_semantic"] = labels.map(label2cid).fillna(0).astype(int)

    clusters = {}
    for lbl, cid in label2cid.items():
        clusters.setdefault(cid, []).append(lbl)

    names = {}
    for cid, mem in clusters.items():
        try:
            names[cid] = tfidf_cluster_label(mem)
        except:
            names[cid] = " / ".join(mem[:3])

    df2[f"{id_col}_semantic_name"] = df2[f"{id_col}_semantic"].map(names).astype(str)
    return df2


# ============================================================
# Split helper (embeddings)
# ============================================================

def split_cluster_embeddings(labels: list[str], k: int):

    emb = build_embeddings_for_labels(labels)

    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    raw = km.fit_predict(emb)

    # The return is simply an array of group indices (0..k-1)
    return raw
