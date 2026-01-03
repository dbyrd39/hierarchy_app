# core/category_layer.py

"""
Category layer utilities.

The category layer is the base text layer from which
semantic layers are built:

    Semantic Layer 0
        ↑
    Semantic Layer 1
        ↑
    Category Layer (this module)
        ↑
    Raw item records

This module provides small, focused helpers to:

  - Normalize and standardize the category column
  - Optionally assign integer category IDs
  - Produce simple summaries of the category layer
"""

# Type hints
from __future__ import annotations
from typing import Optional, List

# External dependencies
import pandas as pd

# Internal dependencies
from .attribute_layer import (
    assign_all_clusters,
    make_cluster_names,
)


# ============================================================
#      Normalization: create a canonical `category_name`
# ============================================================

def ensure_category_name_column(
    df: pd.DataFrame,
    category_col: str,
    *,
    strip: bool = True,
) -> pd.DataFrame:
    """
    Ensure the dataframe has a canonical `category_name` column, derived
    from the specified `category_col`.

    This helper is intentionally simple. The HierarchyEngine will call a
    similar normalization internally, but this function is useful if you
    want to perform category-level analysis or debugging outside of the
    engine.

    Parameters
    ----------
    df :
        Input dataframe containing a product category column.
    category_col :
        Name of the column to treat as the category text source.
    strip :
        If True, strip leading/trailing whitespace from category strings.

    Returns
    -------
    pd.DataFrame
        A copy of the dataframe with a `category_name` column.
    """
    if category_col not in df.columns:
        raise ValueError(f"Category column '{category_col}' not found in dataframe.")

    df_out = df.copy()
    cat = df_out[category_col].astype(str)

    if strip:
        cat = cat.str.strip()

    df_out["category_name"] = cat

    return df_out

def ensure_or_generate_category_name(
    df: pd.DataFrame,
    category_col: Optional[str],
    *,
    extra_excluded_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Ensure the dataframe has a canonical `category_name` column.

    - If `category_col` is provided and exists in df:
        behave like `ensure_category_name_column`.
    - If `category_col` is None (or missing from df):
        automatically generate a synthetic `category_name` by
        clustering rows using attribute sparsity patterns and naming
        those clusters with the attribute-layer naming logic.

    Parameters
    ----------
    df :
        Input dataframe containing a product category column.
    category_col :
        Name of the column to treat as the category text source.
    extra_excluded_cols :
        List of column names to exclude from clustering when auto-generating categories.

    Returns
    -------
    pd.DataFrame
        A copy of the dataframe with a `category_name` column.
    """
    df_out = df.copy()

    # --------------------------------------------------------
    # Case 1: user chose a real category column
    # --------------------------------------------------------
    if category_col is not None and category_col in df_out.columns:
        cat = df_out[category_col].astype(str).str.strip()
        df_out["category_name"] = cat
        return df_out

    # --------------------------------------------------------
    # Case 2: auto-generate synthetic categories
    # --------------------------------------------------------

    # Temporarily treat the whole dataset as one "pseudo-category"
    df_temp = df_out.copy()
    df_temp["category_name"] = "ALL"

    # Use existing attribute-layer code to find subclusters
    df_temp = assign_all_clusters(
        df_temp,
        random_state=42,
        extra_excluded_cols=extra_excluded_cols,
    )

    # Name those subclusters using the same naming logic
    _, df_named = make_cluster_names(
        df_temp,
        extra_excluded_cols=extra_excluded_cols,
    )

    # Our synthetic categories are exactly those subcluster names
    df_out["category_name"] = df_named["attribute_cluster_name"]
    return df_out

# ============================================================
#      Optional: integer category IDs for convenience
# ============================================================

# def add_category_ids(
#     df: pd.DataFrame,
#     *,
#     category_col: str = "category_name",
#     id_col: str = "category_id",
# ) -> pd.DataFrame:
#     """
#     Add a simple, deterministic integer `category_id` column derived from
#     the distinct values in `category_col`.

#     This is NOT required by the HierarchyEngine, but can be convenient
#     for analysis or exports.

#     Parameters
#     ----------
#     df :
#         Input dataframe.
#     category_col :
#         Column holding the category text (typically 'category_name').
#     id_col :
#         Name of the integer ID column to create.

#     Returns
#     -------
#     pd.DataFrame
#         A copy of df with an additional integer id column.
#     """
#     if category_col not in df.columns:
#         raise ValueError(f"Column '{category_col}' not found in dataframe.")

#     df_out = df.copy()
#     cats = df_out[category_col].astype(str).fillna("")

#     unique_cats = sorted(set(cats))
#     cat_to_id = {cat: idx for idx, cat in enumerate(unique_cats)}

#     df_out[id_col] = cats.map(cat_to_id).astype(int)
#     return df_out


# ============================================================
#      Category-level summaries & diagnostics
# ============================================================

# def summarize_categories(
#     df: pd.DataFrame,
#     *,
#     category_col: str = "category_name",
#     example_cols: Optional[List[str]] = None,
#     max_examples: int = 3,
# ) -> pd.DataFrame:
#     """
#     Produce a simple summary table for the category layer, showing:

#       - category_name
#       - count of rows in that category
#       - optional example values from selected columns

#     This is useful for sanity-checking the base layer before or after
#     semantic / attribute clustering.

#     Parameters
#     ----------
#     df :
#         Input dataframe.
#     category_col :
#         Column holding category text (default 'category_name').
#     example_cols :
#         Optional list of columns from which example values will be
#         sampled and concatenated as strings.
#     max_examples :
#         Maximum number of distinct example values to include per category.

#     Returns
#     -------
#     pd.DataFrame
#         A dataframe with one row per category and summary information.
#     """
#     if category_col not in df.columns:
#         raise ValueError(f"Column '{category_col}' not found in dataframe.")

#     df_cat = df.copy()
#     df_cat[category_col] = df_cat[category_col].astype(str)

#     # Base counts
#     grouped = df_cat.groupby(category_col, dropna=False)
#     summary = grouped.size().reset_index(name="n_rows")

#     # Example values per category
#     if example_cols:
#         for col in example_cols:
#             if col not in df_cat.columns:
#                 continue

#             examples = (
#                 grouped[col]
#                 .apply(
#                     lambda s: ", ".join(
#                         sorted(set(map(str, s.dropna().unique())))[:max_examples]
#                     )
#                 )
#                 .reset_index(name=f"{col}_examples")
#             )

#             summary = summary.merge(examples, on=category_col, how="left")

#     summary = summary.sort_values("n_rows", ascending=False).reset_index(drop=True)
#     return summary
