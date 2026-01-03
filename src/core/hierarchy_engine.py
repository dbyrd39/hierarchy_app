# core/hierarchy_engine.py

from __future__ import annotations
from typing import Dict, Optional, List

import pandas as pd
import numpy as np

from .semantic_layer import build_semantic_layer
from .attribute_layer import assign_all_clusters, make_cluster_names
from .category_layer import ensure_or_generate_category_name

class HierarchyEngine:
    """
    Unified BL3 backend engine.

    Responsibilities:
        - Build Category → Semantic Layer 1 → Semantic Layer 0
        - Support rename / reassign / merge / split operations
        - Maintain a consistent hierarchy dataframe
        - Provide semantic layers to the UI
        - Produce attribute-layer clustering and expose it to the UI
    """

    # ============================================================
    # Initialization
    # ============================================================

    def __init__(
        self,
        df: pd.DataFrame,
        category_col: Optional[str],
        attribute_method: str = "sparsity",
        attribute_excluded_cols: Optional[List[str]] = None,
    ):
        df = df.copy()

        # Ensure or generate `category_name`
        df = ensure_or_generate_category_name(
            df,
            category_col,
            extra_excluded_cols=attribute_excluded_cols,
        )

        # Base dataframe (full hierarchy, including attribute layer later)
        self._hier_df: pd.DataFrame = df

        # Cached semantic and attribute-layer views
        self._semantic_layer_1: pd.DataFrame | None = None
        self._semantic_layer_0: pd.DataFrame | None = None
        self._attribute_layer_df: pd.DataFrame | None = None

        # Attribute layer configuration
        self._attribute_excluded_cols: list[str] | None = (
            list(attribute_excluded_cols) if attribute_excluded_cols else None
        )
        self._attribute_method: str = attribute_method  # "sparsity" or "value"

        # Build semantic layers
        self._build_semantic_layers()


    # ============================================================
    # Build semantic layers
    # ============================================================

    def _build_semantic_layers(self) -> None:
        """
        category_name → level_1 (Semantic Layer 1) → level_0 (Semantic Layer 0)
        """
        # -----------------------------
        # Semantic Layer 1 (level 1)
        # -----------------------------
        df1 = build_semantic_layer(
            self._hier_df,
            input_label_col="category_name",
            n_clusters=None,
            output_prefix="level_1",
        )

        # -----------------------------
        # Semantic Layer 0 (level 0)
        # -----------------------------
        df0 = build_semantic_layer(
            df1,
            input_label_col="level_1_name",
            n_clusters=None,
            output_prefix="level_0",
        )

        self._hier_df = df0

        # Extract unique label tables for each level
        self._semantic_layer_1 = self._extract_semantic_layer(1)
        self._semantic_layer_0 = self._extract_semantic_layer(0)

    def _extract_semantic_layer(self, level: int) -> pd.DataFrame:
        """
        level=1 → [category_name, level_1_id, level_1_name]
        level=0 → [level_1_name, level_0_id, level_0_name]
        """
        if level == 1:
            df = self._hier_df[["category_name", "level_1_id", "level_1_name"]]
            df = df.drop_duplicates().sort_values("level_1_id")
            df.columns = ["label", "id", "name"]
            return df.reset_index(drop=True)

        if level == 0:
            df = self._hier_df[["level_1_name", "level_0_id", "level_0_name"]]
            df = df.drop_duplicates().sort_values("level_0_id")
            df.columns = ["label", "id", "name"]
            return df.reset_index(drop=True)

        raise ValueError("Semantic level must be 0 or 1")

    # ============================================================
    # Public API: Get semantic layers
    # ============================================================

    def get_semantic_layer(self, level: int) -> pd.DataFrame:
        if level == 1:
            return self._semantic_layer_1.copy()
        if level == 0:
            return self._semantic_layer_0.copy()
        raise ValueError("Semantic level must be 0 or 1")

    # ============================================================
    # Public API: Apply rename & reassignment
    # ============================================================

    def apply_semantic_changes(
        self,
        level: int,
        rename_map: dict[int, str],
        reassignment_map: dict[str, int],
    ) -> None:
        df_layer = self.get_semantic_layer(level)

        # -------------------------------------
        # Rename clusters
        # -------------------------------------
        for cid, new_name in rename_map.items():
            df_layer.loc[df_layer["id"] == cid, "name"] = new_name

        # -------------------------------------
        # Move items
        # -------------------------------------
        for label, new_cid in reassignment_map.items():
            df_layer.loc[df_layer["label"] == label, "id"] = new_cid

        # -------------------------------------
        # Cleanup ids (ensure ascending 1..K)
        # -------------------------------------
        df_layer = self._cleanup_semantic_layer_ids(df_layer)

        # -------------------------------------
        # Push changes into hierarchy_df
        # -------------------------------------
        self._apply_layer_back_to_hierarchy(level, df_layer)

        # -------------------------------------
        # Re-store layer
        # -------------------------------------
        if level == 1:
            self._semantic_layer_1 = df_layer
        else:
            self._semantic_layer_0 = df_layer

    # ============================================================
    # Apply updated semantic layer back to hierarchy DF
    # ============================================================

    def _apply_layer_back_to_hierarchy(self, level: int, df_layer: pd.DataFrame) -> None:
        if level == 1:
            # Map category_name → new ids/names
            m = df_layer.set_index("label")[["id", "name"]].to_dict(orient="index")
            self._hier_df["level_1_id"] = self._hier_df["category_name"].map(
                lambda x: m[x]["id"]
            )
            self._hier_df["level_1_name"] = self._hier_df["category_name"].map(
                lambda x: m[x]["name"]
            )

            # Level 0 depends on level 1 labels → must rebuild
            self._hier_df = build_semantic_layer(
                self._hier_df,
                input_label_col="level_1_name",
                n_clusters=None,
                output_prefix="level_0",
            )
            self._semantic_layer_0 = self._extract_semantic_layer(0)

        else:
            # Level 0 maps from level_1_name
            m = df_layer.set_index("label")[["id", "name"]].to_dict(orient="index")
            self._hier_df["level_0_id"] = self._hier_df["level_1_name"].map(
                lambda x: m[x]["id"]
            )
            self._hier_df["level_0_name"] = self._hier_df["level_1_name"].map(
                lambda x: m[x]["name"]
            )

        # Attribute layer cache is now stale (if it existed)
        self._attribute_layer_df = None

    # ============================================================
    # Cleanup: ensure contiguous IDs
    # ============================================================

    def _cleanup_semantic_layer_ids(self, df_layer: pd.DataFrame) -> pd.DataFrame:
        """
        Ensures cluster IDs become 1..K in ascending order.
        """
        unique_ids = sorted(df_layer["id"].unique())
        remap = {old: i + 1 for i, old in enumerate(unique_ids)}
        df_layer["id"] = df_layer["id"].map(remap)
        return df_layer

    # ============================================================
    # Public API: Recluster semantic layers
    # ============================================================

    def semantic_recluster(self, level: int) -> None:
        """
        Rebuild a semantic layer from scratch using embeddings.
        If level=1: rebuild both level 1 and level 0.
        If level=0: rebuild level 0 only.
        """
        if level == 1:
            df1 = build_semantic_layer(
                self._hier_df,
                input_label_col="category_name",
                n_clusters=None,
                output_prefix="level_1",
            )
            self._hier_df = df1
            self._semantic_layer_1 = self._extract_semantic_layer(1)

            df0 = build_semantic_layer(
                self._hier_df,
                input_label_col="level_1_name",
                n_clusters=None,
                output_prefix="level_0",
            )
            self._hier_df = df0
            self._semantic_layer_0 = self._extract_semantic_layer(0)
        else:
            df0 = build_semantic_layer(
                self._hier_df,
                input_label_col="level_1_name",
                n_clusters=None,
                output_prefix="level_0",
            )
            self._hier_df = df0
            self._semantic_layer_0 = self._extract_semantic_layer(0)

        # Semantic changes invalidate attribute layer cache
        self._attribute_layer_df = None

    # ============================================================
    # Public API: Merge semantic clusters
    # ============================================================

    def merge_semantic_clusters(self, level: int, from_cluster: int, to_cluster: int) -> None:
        df_layer = self.get_semantic_layer(level)

        # Update IDs
        df_layer.loc[df_layer["id"] == from_cluster, "id"] = to_cluster

        # Clean IDs
        df_layer = self._cleanup_semantic_layer_ids(df_layer)

        # Write back to hierarchy
        self._apply_layer_back_to_hierarchy(level, df_layer)

        # Store updated layer
        if level == 1:
            self._semantic_layer_1 = df_layer
        else:
            self._semantic_layer_0 = df_layer

    # ============================================================
    # Public API: Split semantic clusters (MANUAL)
    # ============================================================

    def split_semantic_cluster(
        self,
        level: int,
        from_cluster: int,
        labels_to_move: list[str],
    ) -> None:
        """
        Manually split a semantic cluster by moving selected labels
        into a NEW cluster.

        - level: 0 or 1
        - from_cluster: cluster ID to split
        - labels_to_move: list of labels (strings) currently in from_cluster
                          that should be peeled off into a new cluster.

        New cluster gets a fresh ID (max+1), then IDs are normalized (1..K).
        """
        df_layer = self.get_semantic_layer(level)

        if not labels_to_move:
            raise ValueError("No labels provided to move.")

        # Short aliases
        label_col = "label"
        id_col = "id"

        # Validate labels belong to from_cluster
        in_cluster_mask = df_layer[id_col] == from_cluster
        labels_in_cluster = set(df_layer.loc[in_cluster_mask, label_col].tolist())

        invalid = [lbl for lbl in labels_to_move if lbl not in labels_in_cluster]
        if invalid:
            raise ValueError(
                f"Cannot split: these labels are not in cluster {from_cluster}: {invalid}"
            )

        if len(labels_in_cluster) == len(labels_to_move):
            raise ValueError(
                "Cannot move all items out of the cluster. At least one item must remain."
            )

        # Assign a fresh new cluster ID (Option A)
        max_id = int(df_layer[id_col].max())
        new_id = max_id + 1

        df_layer.loc[df_layer[label_col].isin(labels_to_move), id_col] = new_id

        # Cleanup & write back
        df_layer = self._cleanup_semantic_layer_ids(df_layer)
        self._apply_layer_back_to_hierarchy(level, df_layer)

        if level == 1:
            self._semantic_layer_1 = df_layer
        else:
            self._semantic_layer_0 = df_layer

    # ============================================================
    # Attribute layer: build + cache
    # ============================================================

    def _build_attribute_layer(self) -> None:
        """
        Build attribute-based clusters using the attribute_layer utilities.

        We:
        - Run assign_all_subclusters to compute `category_subcluster`
        - Use make_subcluster_names_tfidf to generate names
        - Expose them as attribute_cluster_id / attribute_cluster_name
        """
        df = self._hier_df.copy()

        excluded = getattr(self, "_attribute_excluded_cols", None)
        method = getattr(self, "_attribute_method", "sparsity")

        # 1) Assign subclusters within each category (attribute-based)
        df = assign_all_clusters(
            df,
            random_state=42,
            extra_excluded_cols=excluded,
            method=method,
        )

        # 2) Name subclusters
        _, df_named = make_cluster_names(
            df,
            extra_excluded_cols=excluded,
        )

        # 3) Expose subcluster info as attribute-layer columns
        df_named["attribute_cluster_id"] = df_named["category_subcluster"]
        df_named["attribute_cluster_name"] = df_named["category_subcluster_name"]

        # Cache full hierarchy + attribute-layer view
        self._hier_df = df_named
        self._attribute_layer_df = (
            df_named[
                [
                    "category_name",
                    "attribute_cluster_id",
                    "attribute_cluster_name",
                ]
            ]
            .drop_duplicates()
            .reset_index(drop=True)
        )



    # ============================================================
    # Public API: Attribute layer getters / mutators
    # ============================================================

    def get_attribute_layer(self) -> pd.DataFrame:
        """
        Returns a dataframe for the attribute layer:
            category_name, attribute_cluster_id, attribute_cluster_name
        """
        if self._attribute_layer_df is None:
            self._build_attribute_layer()
        return self._attribute_layer_df.copy()
    
    def get_attribute_method(self) -> str:
        """
        Returns the currently configured attribute clustering method:
        'sparsity' or 'value'.
        """
        return getattr(self, "_attribute_method", "sparsity")
    

    def attribute_recluster(self, method: str | None = None) -> None:
        """
        Re-run attribute-based clustering from scratch.

        Parameters
        ----------
        method :
            'sparsity' or 'value'. If None, reuse the current method.
        """
        if method is not None:
            self._attribute_method = method

        self._attribute_layer_df = None
        self._build_attribute_layer()


    def apply_attribute_changes(
        self,
        rename_map: Dict[int, str],
        reassignment_map: Dict[str, int],
    ) -> None:
        """
        Apply user edits to the attribute layer:

        - rename_map: attribute_cluster_id → new name
        - reassignment_map: category_name → new attribute_cluster_id

        Note: reassignment by category_name updates all rows belonging
        to that category to the new attribute cluster.
        """
        if self._attribute_layer_df is None:
            self._build_attribute_layer()

        df = self._hier_df.copy()
        layer_df = self._attribute_layer_df.copy()

        # 1) Reassign categories → new attribute_cluster_id
        for cat_name, new_cid in reassignment_map.items():
            new_cid = int(new_cid)

            mask_h = df["category_name"] == cat_name
            df.loc[mask_h, "attribute_cluster_id"] = new_cid

            mask_l = layer_df["category_name"] == cat_name
            layer_df.loc[mask_l, "attribute_cluster_id"] = new_cid

        # 2) Apply renames: attribute_cluster_id → new name
        for cid, new_name in rename_map.items():
            cid = int(cid)

            mask_l = layer_df["attribute_cluster_id"] == cid
            layer_df.loc[mask_l, "attribute_cluster_name"] = new_name

            mask_h = df["attribute_cluster_id"] == cid
            df.loc[mask_h, "attribute_cluster_name"] = new_name

        # 3) Save back
        self._hier_df = df
        self._attribute_layer_df = (
            layer_df[
                [
                    "category_name",
                    "attribute_cluster_id",
                    "attribute_cluster_name",
                ]
            ]
            .drop_duplicates()
            .reset_index(drop=True)
        )

    def set_attribute_excluded_columns(self, cols: list[str] | None) -> None:
        """
        Set additional columns to exclude from attribute clustering and naming.
        Passing None or [] resets to default behavior.
        """
        if cols is None:
            self._attribute_excluded_cols = None
        else:
            # Store as simple list of strings
            self._attribute_excluded_cols = [str(c) for c in cols]

        # Changing exclusions invalidates any cached attribute layer
        self._attribute_layer_df = None


    # ============================================================
    # Public API: Final summarizing
    # ============================================================

    def get_hierarchy_df(self) -> pd.DataFrame:
        return self._hier_df.copy()
