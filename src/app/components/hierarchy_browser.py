import streamlit as st
import pandas as pd
from typing import Optional, List


# ============================================================
#                 Hierarchy Browser Component
# ============================================================

class HierarchyBrowser:
    """
    Interactive browser for the final hierarchy dataframe.

    Expected column structure:

        level_0_id
        level_0_name
        level_1_id
        level_1_name
        category_name
        attribute_cluster_id          # attribute-level cluster id
        attribute_cluster_name        # attribute-level cluster name
        ... plus any detail columns

    The browser behaves like a file-system explorer:
        Semantic Layer 0 → Semantic Layer 1 → Category → Attribute Layer → Individual Rows
    """

    def __init__(self, df_hierarchy: pd.DataFrame):
        self.df = df_hierarchy

    # --------------------------------------------------------
    # Helpers
    # --------------------------------------------------------

    def _guess_name_col(self) -> Optional[str]:
        """
        Guesses a reasonable name/description column for display.
        """
        candidates = [
            "product_name",
            "item_name",
            "description",
            "product_description",
        ]
        for c in candidates:
            if c in self.df.columns:
                return c

        # fallback: first non-hierarchy text column
        hierarchy_cols = {
            "level_0_id",
            "level_0_name",
            "level_1_id",
            "level_1_name",
            "category_name",
            "attribute_cluster_id",
            "attribute_cluster_name",
        }
        for col in self.df.columns:
            if col not in hierarchy_cols and self.df[col].dtype == object:
                return col

        return None

    def _check_columns(self) -> bool:
        """
        Check if we have at least the key semantic/category columns.
        Returns True if we can render a hierarchical view.
        """
        required = ["level_0_id", "level_0_name", "level_1_id", "level_1_name", "category_name"]
        missing = [c for c in required if c not in self.df.columns]
        if missing:
            st.warning(
                "Some required hierarchy columns are missing: "
                + ", ".join(missing)
                + ".\nShowing raw dataframe instead."
            )
            st.dataframe(self.df.head(200), use_container_width=True)
            return False
        return True

    # --------------------------------------------------------
    # Render
    # --------------------------------------------------------

    def render(self):
        """
        Renders the hierarchy browser UI.
        """
        if not isinstance(self.df, pd.DataFrame) or self.df.empty:
            st.warning("Hierarchy dataframe is empty or invalid.")
            return

        if not self._check_columns():
            return

        row_name_col = self._guess_name_col()

        # ----------------------------------------------------
        # Level 0 selection (Semantic Layer 0)
        # ----------------------------------------------------
        st.markdown("### Semantic Layer 0 Selection")

        level0_options = (
            self.df[["level_0_id", "level_0_name"]]
            .drop_duplicates()
            .sort_values("level_0_id")
        )
        level0_dict = {
            f"{int(row.level_0_id)} — {row.level_0_name}": int(row.level_0_id)
            for _, row in level0_options.iterrows()
        }

        if not level0_dict:
            st.warning("No Semantic Layer 0 clusters found.")
            st.dataframe(self.df.head(200), use_container_width=True)
            return

        selected_level0_label = st.selectbox(
            "Select a Semantic Layer 0 cluster:",
            options=list(level0_dict.keys()),
        )
        selected_level0_id = level0_dict[selected_level0_label]

        df_l0 = self.df[self.df["level_0_id"] == selected_level0_id]

        # ----------------------------------------------------
        # Level 1 selection (Semantic Layer 1)
        # ----------------------------------------------------
        st.markdown("### Semantic Layer 1 Selection")

        level1_options = (
            df_l0[["level_1_id", "level_1_name"]]
            .drop_duplicates()
            .sort_values("level_1_id")
        )
        level1_dict = {
            f"{int(row.level_1_id)} — {row.level_1_name}": int(row.level_1_id)
            for _, row in level1_options.iterrows()
        }

        if not level1_dict:
            st.warning(
                "No Semantic Layer 1 clusters found under the selected Semantic Layer 0."
            )
            st.dataframe(df_l0.head(200), use_container_width=True)
            return

        selected_level1_label = st.selectbox(
            "Select a Semantic Layer 1 cluster:",
            options=list(level1_dict.keys()),
        )
        selected_level1_id = level1_dict[selected_level1_label]

        df_l1 = df_l0[df_l0["level_1_id"] == selected_level1_id]

        # ----------------------------------------------------
        # Category selection
        # ----------------------------------------------------
        st.markdown("### Category Selection")

        category_options = (
            df_l1[["category_name"]]
            .drop_duplicates()
            .sort_values("category_name")
        )
        category_list = category_options["category_name"].tolist()

        if not category_list:
            st.warning(
                "No categories found under the selected Semantic Layer 1 cluster."
            )
            st.dataframe(df_l1.head(200), use_container_width=True)
            return

        selected_category = st.selectbox(
            "Select a Category:",
            options=category_list,
        )

        df_cat = df_l1[df_l1["category_name"] == selected_category]

        # ----------------------------------------------------
        # Attribute Layer selection (if present)
        # ----------------------------------------------------
        has_attribute_layer = (
            "attribute_cluster_id" in self.df.columns
            and "attribute_cluster_name" in self.df.columns
        )

        selected_attr_id = None
        df_attr = df_cat

        if has_attribute_layer:
            st.markdown("### Attribute Layer Selection")

            attr_options = (
                df_cat[["attribute_cluster_id", "attribute_cluster_name"]]
                .drop_duplicates()
                .sort_values("attribute_cluster_id")
            )

            if attr_options.empty:
                st.info(
                    "No attribute clusters found for this category. "
                    "Showing all rows under the category."
                )
            else:
                attr_dict = {
                    f"{int(row.attribute_cluster_id)} — {row.attribute_cluster_name}":
                        int(row.attribute_cluster_id)
                    for _, row in attr_options.iterrows()
                }

                selected_attr_label = st.selectbox(
                    "Select an Attribute Cluster (Attribute Layer):",
                    options=list(attr_dict.keys()),
                )
                selected_attr_id = attr_dict[selected_attr_label]
                df_attr = df_cat[df_cat["attribute_cluster_id"] == selected_attr_id]

        # ----------------------------------------------------
        # Final rows table
        # ----------------------------------------------------
        st.markdown("### Rows in Current Selection")

        max_rows = 500
        df_leaf = df_attr.copy()

        display_cols: List[str] = []

        if row_name_col and row_name_col in df_leaf.columns:
            display_cols.append(row_name_col)

        for col in [
            "level_0_name",
            "level_1_name",
            "category_name",
            "attribute_cluster_name",
        ]:
            if col in df_leaf.columns and col not in display_cols:
                display_cols.append(col)

        if not display_cols:
            display_cols = list(df_leaf.columns)

        st.dataframe(
            df_leaf[display_cols].reset_index(drop=True).head(max_rows),
            use_container_width=True,
        )

        if len(df_leaf) > max_rows:
            st.caption(
                f"Showing first {max_rows} rows out of {len(df_leaf)} products."
            )

        # Path display
        st.markdown("---")
        path_parts = [selected_level0_label, selected_level1_label, selected_category]
        if has_attribute_layer and selected_attr_id is not None:
            # find attribute name
            attr_name = (
                df_attr["attribute_cluster_name"].iloc[0]
                if "attribute_cluster_name" in df_attr.columns and not df_attr.empty
                else f"Attribute Cluster {selected_attr_id}"
            )
            path_parts.append(attr_name)

        st.markdown(
            "**Current Path:** " + " ▸ ".join(path_parts)
        )
