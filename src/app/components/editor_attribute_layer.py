import streamlit as st
import pandas as pd
from typing import Dict


# ============================================================
#             Attribute Layer Editor Component
# ============================================================


class AttributeLayerEditor:
    """
    UI component for editing the attribute layer.

    Expects the backend engine to provide:

        engine.get_attribute_layer() -> pd.DataFrame
            Columns:
                - category_name
                - attribute_cluster_id
                - attribute_cluster_name

        engine.attribute_recluster(method="sparsity")

        engine.apply_attribute_changes(rename_map, reassignment_map)
            rename_map:        {attribute_cluster_id -> new_name}
            reassignment_map:  {category_name -> new_attribute_cluster_id}
                               (we'll send an empty dict now, since the UI
                                no longer exposes moves)
    """

    def __init__(
        self,
        wizard,
        engine,
        title: str = "Attribute Layer Editor",
        method: str = "sparsity",
    ):
        self.wizard = wizard
        self.engine = engine
        self.title = title
        self.method = method  # 'sparsity' or 'value'


    # ------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------

    def _load_layer(self) -> pd.DataFrame:
        try:
            return self.engine.get_attribute_layer()
        except Exception as e:
            st.error(f"Failed to load attribute layer: {e}")
            return pd.DataFrame()

    # ----------------------- Rename section ---------------------

    def _render_rename_section(self, df_layer: pd.DataFrame) -> Dict[int, str]:
        st.markdown("### Rename Attribute Clusters")

        rename_map: Dict[int, str] = {}

        if df_layer.empty:
            st.info("No attribute clusters to rename.")
            return rename_map

        # Expect these columns as produced by HierarchyEngine._build_attribute_layer
        id_col = "attribute_cluster_id"
        name_col = "attribute_cluster_name"

        if id_col not in df_layer.columns or name_col not in df_layer.columns:
            st.error(
                f"Expected columns '{id_col}' and '{name_col}' in attribute layer dataframe."
            )
            return rename_map

        cluster_ids = sorted(df_layer[id_col].unique())
        selected_cid = st.selectbox(
            "Select attribute cluster to rename",
            options=cluster_ids,
            key="attr_rename_select",
        )

        current_name = (
            df_layer.loc[df_layer[id_col] == selected_cid, name_col]
            .iloc[0]
        )
        st.markdown(f"Current name: **{current_name}**")

        new_name = st.text_input(
            "New attribute cluster name",
            key="attr_rename_new",
        )

        if new_name.strip() and new_name.strip() != current_name:
            rename_map[int(selected_cid)] = new_name.strip()

        return rename_map

    # ------------------------------------------------------------
    # Render
    # ------------------------------------------------------------

    def render(self) -> None:
        st.markdown(f"## {self.title}")

        if self.engine is None:
            st.error("Hierarchy engine not initialized.")
            return

        df_layer = self._load_layer()
        if df_layer.empty:
            st.warning("Attribute layer is currently empty.")
            # Still allow recompute
            if st.button(
                "Recompute Attribute Clusters",
                type="secondary",
                key="btn_attr_recluster_empty",
            ):
                try:
                    self.engine.attribute_recluster(method=self.method)
                    st.success("Recomputed attribute clusters.")
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Failed to recompute attribute clusters: {e}")
            return

        st.dataframe(df_layer, use_container_width=True)

        st.markdown("---")

        # Recompute attribute layer
        if st.button(
            "Recompute Attribute Clusters",
            type="secondary",
            key="btn_attr_recluster",
        ):
            try:
                self.engine.attribute_recluster(method=self.method)
                st.success("Recomputed attribute clusters.")
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Failed to recompute attribute clusters: {e}")

        st.markdown("---")

        # Rename only (no move/reassign section)
        rename_map = self._render_rename_section(df_layer)

        st.markdown("---")

        # Apply changes (reassignment_map is always empty for now)
        if st.button(
            "Apply Attribute Layer Changes",
            type="primary",
            key="btn_attr_apply_changes",
        ):
            try:
                self.engine.apply_attribute_changes(
                    rename_map=rename_map,
                    reassignment_map={},  # no category moves from UI
                )
                st.success("Attribute layer updated.")
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Failed to apply attribute-layer changes: {e}")
