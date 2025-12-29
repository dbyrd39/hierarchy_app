# core/attribute_layer.py

"""
This file implements the Attribute Layer.

The Attribute Layer groups rows within each category based on
their attribute sparsity patterns (which columns are populated), and
produces human-readable names for each attribute-based cluster.

Public API used by the hierarchy engine:

    assign_all_clusters(df, random_state=42) -> pd.DataFrame
        Returns a copy of df with an integer "attribute_cluster"
        column indicating the attribute-layer cluster id for each row.

    make_cluster_names(df) -> (dict, pd.DataFrame)
        Returns (cluster_name_map, df_with_names) where
        "attribute_cluster_name" is added to the dataframe.
"""

# Type hints
from __future__ import annotations
from typing import List, Tuple, Dict

# External dependencies
import numpy as np
import pandas as pd

# Sklearn dependencies
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans


# ------------------------------------------------------------
# Column selection helper
# ------------------------------------------------------------

# Columns to exclude from attribute consideration
_METADATA_COLS = {
    "category_name",
    "category_cluster",
    "category_cluster_name",
    "attribute_cluster_id",
    "attribute_cluster_name",
    "level_0_id",
    "level_0_name",
    "level_1_id",
    "level_1_name",
}

# Function to select attribute columns for clustering
def _select_attribute_columns(
    df: pd.DataFrame,
    extra_excluded_cols: list[str] | None = None,
) -> List[str]:
    """
    Heuristic to choose attribute columns for sparsity-based clustering.

    We treat any non-all-null column that is not obviously part of the
    hierarchy metadata as an attribute candidate, and also allow the caller
    to explicitly exclude additional columns via `extra_excluded_cols`.

    Parameters
    ----------
    df :
        Primary DataFrame object for use in hierarchy generation
    extra_excluded_cols :
        Columns to exclude from clustering

    Returns
    ------- 
    List[str] :
        An list of column names to be used as attribute columns
    """
    excluded = set(_METADATA_COLS)
    if extra_excluded_cols:
        excluded |= set(map(str, extra_excluded_cols))

    cols: List[str] = []
    for col in df.columns:
        if col in excluded:
            continue
        # Skip index-like or trivial columns
        if str(col).lower() in {"index", "id"}:
            continue
        if df[col].isna().all():
            continue
        cols.append(col)
    return cols


# ------------------------------------------------------------
# Core sparsity-based clustering per category
# ------------------------------------------------------------

def _cluster_products_within_category_sparsity(
    cat_df: pd.DataFrame,
    attr_cols: List[str],
    random_state: int = 42,
) -> np.ndarray:
    """
    Cluster products within a single category using attribute sparsity
    (which columns are non-null / non-empty).

    Parameters
    ----------
    cat_df :
        Subset of the dataframe containing only rows for a single category.
    attr_cols :
        Columns to treat as attributes.
    random_state :
        Seed for KMeans and SVD.

    Returns
    -------
    np.ndarray
        An array of integer labels (cluster ids) with length len(cat_df).
    """
    if not attr_cols or cat_df.empty:
        # Degenerate: single cluster
        return np.zeros(len(cat_df), dtype=int)

    # Build a boolean mask: row x attribute_col
    mask = pd.DataFrame(False, index=cat_df.index, columns=attr_cols)

    for col in attr_cols:
        col_series = cat_df[col]
        if col_series.dtype == object:
            mask[col] = col_series.notna() & (col_series.astype(str).str.strip() != "")
        else:
            mask[col] = col_series.notna()

    # Convert to array
    X = mask.to_numpy(dtype=float)
    n_samples, n_features = X.shape

    if n_samples <= 1:
        return np.zeros(n_samples, dtype=int)

    # If all-zero features, give up
    if not np.any(X):
        return np.zeros(n_samples, dtype=int)

    # Determine k using sqrt heuristic, clipped
    k = int(np.clip(np.sqrt(n_samples), 2, 8))
    if k > n_samples:
        k = n_samples
    if k <= 1:
        return np.zeros(n_samples, dtype=int)

    # Dimensionality reduction for very wide data
    n_components = min(20, n_features - 1, n_samples - 1) if n_features > 1 else 1
    if n_components >= 1:
        try:
            svd = TruncatedSVD(n_components=n_components, random_state=random_state)
            X_reduced = svd.fit_transform(X)
        except Exception:
            # Fallback: no reduction
            X_reduced = X
    else:
        X_reduced = X

    # Cluster
    try:
        km = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
        labels = km.fit_predict(X_reduced)
    except Exception:
        # Fallback: single cluster
        labels = np.zeros(n_samples, dtype=int)

    # Normalize labels to 0..C-1 per category
    unique_raw = sorted(set(int(l) for l in labels))
    raw_to_new = {raw: i for i, raw in enumerate(unique_raw)}
    labels_norm = np.array([raw_to_new[int(l)] for l in labels], dtype=int)

    return labels_norm

def _cluster_products_within_category_value(
    cat_df: pd.DataFrame,
    attr_cols: List[str],
    random_state: int = 42,
) -> np.ndarray:
    """
    Cluster products within a single category using attribute VALUES
    (not just presence/absence).

    Parameters
    ----------
    cat_df :
        Subset of the dataframe containing only rows for a single category.
    attr_cols :
        Columns to treat as attributes.
    random_state :
        Seed for KMeans and SVD.

    Returns
    -------
    np.ndarray
        An array of integer labels (cluster ids) with length len(cat_df).
    """
    if not attr_cols or cat_df.empty:
        return np.zeros(len(cat_df), dtype=int)

    features = []
    for col in attr_cols:
        s = cat_df[col]
        if s.isna().all():
            continue

        if pd.api.types.is_numeric_dtype(s):
            filled = s.fillna(s.median())
            features.append(filled.to_numpy(dtype=float).reshape(-1, 1))
        else:
            # Factorize string values
            codes, _ = pd.factorize(s.astype(str), sort=True)
            # Treat -1 (NA) as separate code
            codes = codes.astype(float)
            features.append(codes.reshape(-1, 1))

    if not features:
        return np.zeros(len(cat_df), dtype=int)

    X = np.hstack(features)
    n_samples, n_features = X.shape

    if n_samples <= 1:
        return np.zeros(n_samples, dtype=int)

    if not np.any(np.isfinite(X)):
        return np.zeros(n_samples, dtype=int)

    # Determine k using sqrt heuristic, clipped
    k = int(np.clip(np.sqrt(n_samples), 2, 8))
    if k > n_samples:
        k = n_samples
    if k <= 1:
        return np.zeros(n_samples, dtype=int)

    # Optional dimensionality reduction for wide matrices
    n_components = min(20, n_features - 1, n_samples - 1) if n_features > 1 else 1
    if n_components >= 1:
        try:
            svd = TruncatedSVD(n_components=n_components, random_state=random_state)
            X_reduced = svd.fit_transform(X)
        except Exception:
            X_reduced = X
    else:
        X_reduced = X

    try:
        km = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
        labels = km.fit_predict(X_reduced)
    except Exception:
        labels = np.zeros(n_samples, dtype=int)

    # Normalize labels to 0..C-1 per category
    unique_raw = sorted(set(int(l) for l in labels))
    raw_to_new = {raw: i for i, raw in enumerate(unique_raw)}
    labels_norm = np.array([raw_to_new[int(l)] for l in labels], dtype=int)

    return labels_norm


# ------------------------------------------------------------
# Public: assign_all_clusters
# ------------------------------------------------------------

def assign_all_clusters(
    df: pd.DataFrame,
    random_state: int = 42,
    extra_excluded_cols: list[str] | None = None,
    method: str = "sparsity",
) -> pd.DataFrame:
    """
    Assign attribute-layer clusters within each category.

    For every distinct value of `category_name`, we cluster its products
    using either:

        - 'sparsity': attribute sparsity patterns (which columns are present)
        - 'value':    attribute values (numeric + factorized categorical)

    and assign an integer `attribute_cluster` id (0..K-1 for that category).

    Parameters
    ----------
    df :
        Input dataframe. Must contain `category_name`.
    random_state :
        Random seed for clustering.
    extra_excluded_cols :
        Optional list of column names to exclude from attribute clustering
        (in addition to the built-in metadata exclusions).
    method :
        'sparsity' or 'value'.

    Returns 
    -------
    pd.DataFrame :
        A dataframe of the original dataset with cluster assignments as integers, and a shape of the original dataset plus an additional column.
    """
    if "category_name" not in df.columns:
        raise ValueError("Expected column 'category_name' in dataframe.")

    attr_cols = _select_attribute_columns(df, extra_excluded_cols=extra_excluded_cols)
    df_out = df.copy()

    all_labels = np.zeros(len(df_out), dtype=int)

    for cat, cat_idx in df_out.groupby("category_name").groups.items():
        cat_df = df_out.loc[cat_idx]

        if method == "value":
            labels = _cluster_products_within_category_value(
                cat_df,
                attr_cols,
                random_state=random_state,
            )
        else:
            labels = _cluster_products_within_category_sparsity(
                cat_df,
                attr_cols,
                random_state=random_state,
            )

        all_labels[cat_df.index.to_numpy()] = labels

    df_out["attribute_cluster"] = all_labels
    return df_out



# ------------------------------------------------------------
# Public: make_cluster_names
# ------------------------------------------------------------

def make_cluster_names(
    df: pd.DataFrame,
    purity_threshold: float = 0.5,
    extra_excluded_cols: list[str] | None = None,
) -> Tuple[Dict[tuple, str], pd.DataFrame]:
    """
    Compute human-readable names for attribute-layer clusters.

    For each (category_name, category_cluster) group, we scan the
    attribute columns and look for columns whose values are relatively
    pure within the cluster (most rows share the same non-null value).

    We then generate a label such as:

        "Binding Covers – 210 (Width mm) / 16 (Height mm)"

    where the pieces are taken from high-purity attribute values.

    Parameters
    ----------
    df :
        Input dataframe. Must contain 'category_name' and 'attribute_cluster'.
    purity_threshold :
        Minimum fraction of rows within a cluster that must share the
        same value in an attribute column for that (column, value)
        descriptor to be used in the label.
    extra_excluded_cols :
        Optional list of columns to exclude from naming consideration.

    Returns
    -------
    Tuple[Dict[tuple, str], pd.DataFrame] :
        Category-cluster pair mapped to the corresponding cluster label, coupled with the resulting dataframe that includes the category-cluster name.
    """
    if "category_name" not in df.columns or "attribute_cluster" not in df.columns:
        raise ValueError("Expected 'category_name' and 'attribute_cluster' columns.")

    df_out = df.copy()
    attr_cols = _select_attribute_columns(df_out, extra_excluded_cols=extra_excluded_cols)

    cluster_name_map: Dict[tuple, str] = {}
    df_out["attribute_cluster_name"] = ""

    if not attr_cols:
        # Degenerate: just use category name + generic suffix
        for (cat, cid), idx in df_out.groupby(
            ["category_name", "attribute_cluster"]
        ).groups.items():
            label = f"{cat} – misc"
            cluster_name_map[(cat, cid)] = label
            df_out.loc[idx, "attribute_cluster_name"] = label
        return cluster_name_map, df_out

    # Treat these string values as "missing" when naming
    nan_like = {"nan", "none", "null", "na", "n/a"}

    for (cat, cid), grp in df_out.groupby(["category_name", "attribute_cluster"]):
        n_rows = len(grp)
        descriptors: List[str] = []

        # Scan each attribute column for high-purity values
        for col in attr_cols:
            col_series = grp[col]

            # Skip if all null
            if col_series.isna().all():
                continue

            vals = col_series.astype(str).str.strip()

            # Drop empty strings
            vals = vals[vals != ""]

            # Drop nan-like string representations
            vals = vals[~vals.str.lower().isin(nan_like)]

            if vals.empty:
                continue

            vc = vals.value_counts(dropna=True)
            top_val = vc.index[0]
            top_count = vc.iloc[0]
            purity = top_count / n_rows

            if purity >= purity_threshold:
                descriptors.append(f"{top_val} ({col})")

        if descriptors:
            desc_str = " / ".join(descriptors[:3])
            label = f"{cat} – {desc_str}"
        else:
            label = f"{cat} – misc"

        cluster_name_map[(cat, cid)] = label
        df_out.loc[grp.index, "attribute_cluster_name"] = label

    return cluster_name_map, df_out

