# # core/hierarchy.py

# from __future__ import annotations

# from dataclasses import dataclass
# from typing import Dict, List, Literal, Optional, Any, Tuple

# import math
# import numpy as np
# import pandas as pd
# from sklearn.cluster import KMeans

# #from .category_layer import guess_category_column
# #from .text_utils import build_embeddings_for_labels, tfidf_cluster_label
# from .attribute_layer import (
#     assign_all_subclusters,
#     make_subcluster_names_tfidf,
# )

# from .semantic_layer import build_semantic_layer, semantic_cluster_labels

# # -------------------------
# # Types & configs
# # -------------------------

# HierarchyLayerMethod = Literal[
#     "existing_column",     # use an existing column as-is
#     "semantic_cluster",    # cluster label strings via embeddings
#     "sparsity_cluster",    # TODO: generalize current subcluster logic
# ]

# @dataclass
# class HierarchyLayerConfig:
#     """
#     Generic configuration for one hierarchy level (Level 1, Level 2, ...).

#     We standardize on output columns:
#       - level_{index}_id
#       - level_{index}_name
#     """
#     level_index: int
#     display_name: str
#     method: HierarchyLayerMethod

#     # If method == "existing_column"
#     source_column: Optional[str] = None

#     # If method == "semantic_cluster"
#     semantic_n_clusters: Optional[int] = None
#     semantic_min_per_cluster: int = 5

#     # If method == "sparsity_cluster"
#     sparsity_random_state: int = 42
#     # (You will hook this into your existing subcluster logic later.)

#     extra_params: Dict[str, Any] = None


# # -------------------------
# # Category candidate helper
# # -------------------------

# @dataclass
# class CategoryCandidate:
#     column: str
#     normalized_name: str
#     unique_count: int


# def _normalize_col_name(col: str) -> str:
#     return "".join(ch.lower() for ch in str(col) if ch.isalnum())


# def find_category_candidates(df: pd.DataFrame) -> List[CategoryCandidate]:
#     """
#     Return a list of candidate category-like text columns, excluding
#     numeric / numeric-text columns, sorted broadest -> most granular
#     (by unique_count ascending).
#     """
#     import re
#     numeric_re = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")

#     def is_numeric_text(val: str) -> bool:
#         s = str(val).strip()
#         return bool(numeric_re.match(s))

#     candidates: List[CategoryCandidate] = []

#     for col in df.columns:
#         series = df[col].dropna()

#         # Skip numeric dtypes entirely
#         if pd.api.types.is_numeric_dtype(df[col]):
#             continue

#         # Skip purely numeric text columns
#         if len(series) > 0 and all(is_numeric_text(v) for v in series.astype(str)):
#             continue

#         # Basic uniqueness
#         try:
#             uniq = series.astype(str).nunique()
#         except Exception:
#             continue

#         candidates.append(
#             CategoryCandidate(
#                 column=col,
#                 normalized_name=_normalize_col_name(col),
#                 unique_count=uniq,
#             )
#         )

#     # Sort by unique_count ascending (broadest first)
#     candidates.sort(key=lambda c: c.unique_count)
#     return candidates


# def recommend_num_levels(n_rows: int, num_category_candidates: int) -> int:
#     """
#     Heuristic suggestion for number of hierarchy levels (excluding Root and rows).
#     """
#     if n_rows <= 200:
#         base = 1
#     elif n_rows <= 2000:
#         base = 2
#     elif n_rows <= 20000:
#         base = 3
#     else:
#         base = 4

#     # Nudge up if many category-like columns exist, but cap at 5
#     return min(base + max(0, num_category_candidates - 1), 5)

# # -------------------------
# # Automatic hierarchy planning
# # -------------------------

# @dataclass
# class AutoHierarchyPlan:
#     """
#     Captures the automatically chosen hierarchy plan for a dataset:

#     - layer_configs : the list of HierarchyLayerConfig that build_hierarchy() will use
#     - n_rows        : total number of rows in the dataset
#     - n_unique_cats : unique values in the chosen category column
#     - ratio         : n_unique_cats / n_rows
#     - num_levels    : number of levels actually created (len(layer_configs))
#     - mode          : one of {"few_categories_attr_deep", "mixed", "semantic_only"}
#     """
#     layer_configs: List[HierarchyLayerConfig]
#     n_rows: int
#     n_unique_cats: int
#     ratio: float
#     num_levels: int
#     mode: str


# def make_auto_layer_configs(
#     df: pd.DataFrame,
#     *,
#     category_col: str,
#     max_levels: int = 3,
# ) -> AutoHierarchyPlan:
#     """
#     Automatically choose how many hierarchy levels to build AND which
#     method each level should use, based on:

#       - dataset size
#       - number of unique categories in `category_col`
#       - number of candidate category-like columns

#     Heuristics (conceptual):

#     - If the category column has **very few** distinct values compared to
#       the number of rows, treat the category as a broad "root" concept and
#       build **attribute-based (sparsity)** levels underneath it.

#     - If the category column has a **very high** number of distinct values
#       (close to per-row uniqueness), favor purely **semantic** hierarchy
#       layers and skip attribute-based layers.

#     - Otherwise, build a **mixed** hierarchy:
#         Level 1 / Level 2: semantic over the category labels
#         Level 3 (optional): attribute-based (sparsity) within Level 2.
#     """
#     if category_col not in df.columns:
#         raise KeyError(
#             f"make_auto_layer_configs: category_col '{category_col}' not found in DataFrame."
#         )

#     n_rows = len(df)
#     if n_rows == 0:
#         # Degenerate: no rows → just return 2 trivial semantic layers
#         cfgs = [
#             HierarchyLayerConfig(
#                 level_index=1,
#                 display_name="Level 1 (semantic over category)",
#                 method="semantic_cluster",
#                 source_column=category_col,
#                 semantic_n_clusters=3,
#             ),
#             HierarchyLayerConfig(
#                 level_index=2,
#                 display_name="Level 2 (semantic over Level 1)",
#                 method="semantic_cluster",
#                 semantic_n_clusters=2,
#             ),
#         ]
#         return AutoHierarchyPlan(
#             layer_configs=cfgs,
#             n_rows=0,
#             n_unique_cats=0,
#             ratio=0.0,
#             num_levels=len(cfgs),
#             mode="semantic_only",
#         )

#     cat_series = (
#         df[category_col]
#         .fillna("")
#         .astype(str)
#         .str.strip()
#     )
#     n_unique_cats = cat_series.nunique()
#     ratio = n_unique_cats / max(1, n_rows)

#     # How many category-like columns exist in general?
#     candidates = find_category_candidates(df)
#     num_cat_candidates = len(candidates)

#     # Base suggestion, then clamp to [2, max_levels]
#     suggested_levels = recommend_num_levels(n_rows, num_cat_candidates)
#     num_levels = min(max(2, suggested_levels), max_levels)

#     # Heuristic thresholds
#     LOW_RATIO = 0.01   # e.g. 50 unique categories in 10,000 rows
#     HIGH_RATIO = 0.30  # e.g. 3,000 unique categories in 10,000 rows

#     layer_configs: List[HierarchyLayerConfig] = []

#     # --------- Mode 1: few categories → attribute-based hierarchy ---------
#     if ratio <= LOW_RATIO:
#         mode = "few_categories_attr_deep"

#         for idx in range(1, num_levels + 1):
#             if idx == 1:
#                 # Level 1: attribute-based clustering within the category layer
#                 layer_configs.append(
#                     HierarchyLayerConfig(
#                         level_index=1,
#                         display_name="Level 1 (attribute-based over category)",
#                         method="sparsity_cluster",
#                         source_column=category_col,
#                     )
#                 )
#             else:
#                 # Additional levels: attribute-based over previous level names
#                 layer_configs.append(
#                     HierarchyLayerConfig(
#                         level_index=idx,
#                         display_name=f"Level {idx} (attribute-based over Level {idx - 1})",
#                         method="sparsity_cluster",
#                     )
#                 )

#     # --------- Mode 2: many categories → semantic-only hierarchy ---------
#     elif ratio >= HIGH_RATIO:
#         mode = "semantic_only"

#         # Level 1: semantic over category labels
#         # simple heuristic: sqrt of unique categories, clamped to [3, 50]
#         k1 = max(3, min(50, int(math.sqrt(max(1, n_unique_cats)))))

#         layer_configs.append(
#             HierarchyLayerConfig(
#                 level_index=1,
#                 display_name="Level 1 (semantic over category labels)",
#                 method="semantic_cluster",
#                 source_column=category_col,
#                 semantic_n_clusters=k1,
#             )
#         )

#         if num_levels >= 2:
#             k2 = max(2, int(k1 / 2))
#             layer_configs.append(
#                 HierarchyLayerConfig(
#                     level_index=2,
#                     display_name="Level 2 (semantic over Level 1 clusters)",
#                     method="semantic_cluster",
#                     semantic_n_clusters=k2,
#                 )
#             )

#         if num_levels >= 3:
#             k3 = max(2, int(k2 / 2))
#             layer_configs.append(
#                 HierarchyLayerConfig(
#                     level_index=3,
#                     display_name="Level 3 (semantic over Level 2 clusters)",
#                     method="semantic_cluster",
#                     semantic_n_clusters=k3,
#                 )
#             )

#     # --------- Mode 3: mixed semantic + attribute-based ---------
#     else:
#         mode = "mixed"

#         # Level 1: semantic over category labels
#         k1 = max(3, min(50, int(math.sqrt(max(1, n_unique_cats)))))

#         layer_configs.append(
#             HierarchyLayerConfig(
#                 level_index=1,
#                 display_name="Level 1 (semantic over category labels)",
#                 method="semantic_cluster",
#                 source_column=category_col,
#                 semantic_n_clusters=k1,
#             )
#         )

#         # Level 2: semantic over Level 1 cluster names
#         if num_levels >= 2:
#             k2 = max(2, int(k1 / 2))
#             layer_configs.append(
#                 HierarchyLayerConfig(
#                     level_index=2,
#                     display_name="Level 2 (semantic over Level 1 clusters)",
#                     method="semantic_cluster",
#                     semantic_n_clusters=k2,
#                 )
#             )

#         # Optional Level 3: attribute-based within Level 2
#         if num_levels >= 3:
#             layer_configs.append(
#                 HierarchyLayerConfig(
#                     level_index=3,
#                     display_name="Level 3 (attribute-based within Level 2 clusters)",
#                     method="sparsity_cluster",
#                 )
#             )

#     return AutoHierarchyPlan(
#         layer_configs=layer_configs,
#         n_rows=n_rows,
#         n_unique_cats=n_unique_cats,
#         ratio=ratio,
#         num_levels=len(layer_configs),
#         mode=mode,
#     )




# # -------------------------
# # TODO: Sparsity clustering builder
# # -------------------------

# def build_sparsity_layer(
#     df: pd.DataFrame,
#     *,
#     group_by_col: str,
#     output_prefix: str,
#     random_state: int = 42,
# ) -> pd.DataFrame:
#     """
#     Generalized sparsity-based level, analogous to your current subcluster layer.

#     Semantics:
#       - Use `group_by_col` as the "category" within which we cluster by
#         attribute sparsity.
#       - Internally, we reuse `assign_all_subclusters` + `make_subcluster_names_tfidf`,
#         which expect a column named `category_name` and produce:
#           * category_subcluster (int, local within each category_name)
#           * category_subcluster_name (human-readable label)
#       - We then convert (category_name, category_subcluster) pairs into
#         global IDs and expose them as:

#           {output_prefix}_id
#           {output_prefix}_name
#     """
#     if group_by_col not in df.columns:
#         raise KeyError(
#             f"build_sparsity_layer: group_by_col '{group_by_col}' not found in DataFrame."
#         )

#     # Work on a copy so we don't disturb the original df
#     tmp = df.copy()

#     # We want to treat `group_by_col` as the "category" dimension for this layer.
#     # To safely reuse the existing subcluster logic, which expects `category_name`,
#     # we:
#     #   - drop any existing `category_name` (to avoid ambiguity)
#     #   - create a fresh `category_name` from group_by_col
#     if "category_name" in tmp.columns and group_by_col != "category_name":
#         tmp = tmp.drop(columns=["category_name"])

#     tmp["category_name"] = (
#         df[group_by_col]
#         .fillna("")
#         .astype(str)
#         .str.strip()
#     )

#     # Run your existing sparsity-based subcluster assignment
#     tmp = assign_all_subclusters(tmp, random_state=random_state)

#     # Name each (category_name, category_subcluster)
#     cluster_name_map, tmp = make_subcluster_names_tfidf(tmp)

#     # Build a global cluster ID from (category_name, category_subcluster) pairs
#     pairs = list(
#         zip(
#             tmp["category_name"].astype(str),
#             tmp["category_subcluster"].astype(int),
#         )
#     )
#     codes, uniques = pd.factorize(pairs)

#     # Write results back aligned on the original index
#     df_out = df.copy()
#     df_out[f"{output_prefix}_id"] = pd.Series(codes, index=tmp.index).astype("Int64")
#     df_out[f"{output_prefix}_name"] = pd.Series(
#         tmp["category_subcluster_name"].astype(str),
#         index=tmp.index,
#     )

#     return df_out




# # -------------------------
# # Main hierarchy pipeline
# # -------------------------

# def build_hierarchy(
#     df: pd.DataFrame,
#     layer_configs: List[HierarchyLayerConfig],
# ) -> pd.DataFrame:
#     """
#     Execute a sequence of hierarchy layer configs on df, producing:

#       level_1_id, level_1_name
#       level_2_id, level_2_name
#       ...

#     This is the generic replacement for hard-coded Group/Parent/Subcluster.
#     """
#     df = df.copy()

#     for cfg in layer_configs:
#         prefix = f"level_{cfg.level_index}"

#         if cfg.method == "existing_column":
#             if not cfg.source_column:
#                 raise ValueError(
#                     f"Layer {cfg.level_index} is 'existing_column' but source_column is None."
#                 )
#             col = cfg.source_column
#             df[f"{prefix}_id"] = (
#                 df[col].fillna("").astype(str).str.strip()
#             )
#             df[f"{prefix}_name"] = df[f"{prefix}_id"]

#         elif cfg.method == "semantic_cluster":
#             # Decide what to cluster on:
#             # - For Level 1, we might use source_column (e.g. category_name)
#             # - For Level > 1, we typically cluster the previous level's names
#             if cfg.level_index == 1:
#                 if not cfg.source_column:
#                     raise ValueError(
#                         "First semantic layer must specify source_column."
#                     )
#                 input_col = cfg.source_column
#             else:
#                 input_col = f"level_{cfg.level_index - 1}_name"

#             n_clusters = cfg.semantic_n_clusters or 10

#             df = build_semantic_layer(
#                 df,
#                 input_label_col=input_col,
#                 n_clusters=n_clusters,
#                 output_prefix=prefix,
#             )

#         elif cfg.method == "sparsity_cluster":
#             # For now, this is a placeholder calling the generic sparsity builder
#             # using the previous level name as group key.
#             group_by_col = (
#                 f"level_{cfg.level_index - 1}_name"
#                 if cfg.level_index > 1
#                 else cfg.source_column
#             )

#             df = build_sparsity_layer(
#                 df,
#                 group_by_col=group_by_col,
#                 output_prefix=prefix,
#                 random_state=cfg.sparsity_random_state,
#             )

#         else:
#             raise ValueError(f"Unknown hierarchy layer method: {cfg.method}")

#     return df
