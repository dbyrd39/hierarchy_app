import streamlit as st
import pandas as pd


# ============================================================
#             Semantic Layer Editor Component
# ============================================================


class SemanticLayerEditor:
    """
    UI for editing a semantic layer.

    Supports:
        - Rename clusters (single-cluster rename UX)
        - Move items between clusters
        - Merge clusters
        - Split clusters by manually selecting members to peel off
    """

    def __init__(self, level, wizard):
        self.level = level
        self.wizard = wizard
        self.engine = st.session_state.get("engine", None)

    # ------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------

    def _load_layer(self):
        try:
            return self.engine.get_semantic_layer(self.level)
        except Exception as e:
            st.error(f"Error loading semantic layer {self.level}: {e}")
            return pd.DataFrame()

    # ------------------------------------------------------------
    # Rename clusters (dropdown + single text box)
    # ------------------------------------------------------------

    def _render_rename_section(self, layer_df):
        st.markdown("### Rename Clusters")
        rename_map = {}

        id_col = layer_df.columns[1]
        name_col = layer_df.columns[2]

        cluster_ids = sorted(layer_df[id_col].unique())
        if not cluster_ids:
            st.info("No clusters available to rename.")
            return rename_map

        selected_cid = st.selectbox(
            "Select cluster to rename",
            options=cluster_ids,
            key=f"rename_select_{self.level}",
        )

        # Show current name
        current_name = (
            layer_df.loc[layer_df[id_col] == selected_cid, name_col]
            .iloc[0]
        )
        st.markdown(f"Current name: **{current_name}**")

        new_name = st.text_input(
            "New name",
            key=f"rename_new_{self.level}",
        )

        if new_name.strip() and new_name.strip() != current_name:
            rename_map[selected_cid] = new_name.strip()

        return rename_map

    # ------------------------------------------------------------
    # Move items
    # ------------------------------------------------------------

    def _render_reassignment_section(self, layer_df):
        st.markdown("### Move Items to Another Cluster")

        label_col = layer_df.columns[0]
        id_col = layer_df.columns[1]

        reassignment_map = {}

        all_labels = sorted(layer_df[label_col].unique())
        all_clusters = sorted(layer_df[id_col].unique())

        if not all_labels or not all_clusters:
            st.info("No items or clusters available for reassignment.")
            return reassignment_map

        selected_label = st.selectbox(
            "Item to move",
            options=all_labels,
            key=f"move_label_{self.level}",
        )

        new_cluster = st.selectbox(
            "Move item to cluster",
            options=all_clusters,
            key=f"move_target_cluster_{self.level}",
        )

        if st.button(
            "Apply Move",
            type="secondary",
            key=f"btn_move_apply_{self.level}",
        ):
            reassignment_map[selected_label] = int(new_cluster)
            st.success(f"Moved {selected_label} → cluster {new_cluster}")

        return reassignment_map

    # ------------------------------------------------------------
    # Merge clusters
    # ------------------------------------------------------------

    def _render_merge_section(self, layer_df):
        st.markdown("### Merge Clusters")

        id_col = layer_df.columns[1]
        unique_clusters = sorted(layer_df[id_col].unique())

        if len(unique_clusters) < 2:
            st.info("Need at least two clusters to perform a merge.")
            return

        col1, col2 = st.columns(2)
        with col1:
            merge_from = st.selectbox(
                "Merge FROM cluster",
                options=unique_clusters,
                key=f"merge_from_{self.level}",
            )
        with col2:
            merge_into = st.selectbox(
                "Merge INTO cluster",
                options=[c for c in unique_clusters if c != merge_from],
                key=f"merge_into_{self.level}",
            )

        if st.button(
            "Merge Clusters",
            type="primary",
            key=f"btn_merge_clusters_{self.level}",
        ):
            try:
                self.engine.merge_semantic_clusters(
                    level=self.level,
                    from_cluster=int(merge_from),
                    to_cluster=int(merge_into),
                )
                st.success(f"Merged cluster {merge_from} → {merge_into}")
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Merge failed: {e}")

    # ------------------------------------------------------------
    # Split clusters (manual selection)
    # ------------------------------------------------------------

    def _render_split_section(self, layer_df):
        st.markdown("### Split a Cluster (Manual Selection)")

        label_col = layer_df.columns[0]
        id_col = layer_df.columns[1]

        unique_clusters = sorted(layer_df[id_col].unique())
        if not unique_clusters:
            st.info("No clusters available to split.")
            return

        split_cluster = st.selectbox(
            "Select cluster to split",
            options=unique_clusters,
            key=f"split_cluster_{self.level}",
        )

        # All labels currently in that cluster
        cluster_labels = sorted(
            layer_df.loc[layer_df[id_col] == split_cluster, label_col].unique()
        )
        if len(cluster_labels) <= 1:
            st.info("Selected cluster has 1 or fewer items; cannot split.")
            return

        selected_labels = st.multiselect(
            "Select items to move into a NEW cluster",
            options=cluster_labels,
            key=f"split_labels_{self.level}",
        )

        if st.button(
            "Create New Cluster From Selected Items",
            type="primary",
            key=f"btn_split_cluster_{self.level}",
        ):
            if not selected_labels:
                st.warning("Please select at least one item to split off.")
                return
            if len(selected_labels) == len(cluster_labels):
                st.warning(
                    "Cannot move all items out of the cluster. "
                    "At least one item must remain."
                )
                return

            try:
                self.engine.split_semantic_cluster(
                    level=self.level,
                    from_cluster=int(split_cluster),
                    labels_to_move=list(selected_labels),
                )
                st.success(
                    f"Created a new cluster from {len(selected_labels)} selected item(s)."
                )
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Split failed: {e}")

    # ------------------------------------------------------------
    # Main render
    # ------------------------------------------------------------

    def render(self):
        st.markdown(f"### Semantic Layer {self.level}")

        if self.engine is None:
            st.error("Hierarchy engine not initialized.")
            return

        layer_df = self._load_layer()
        if layer_df.empty:
            st.warning("Empty layer.")
            return

        st.dataframe(layer_df, use_container_width=True)

        st.markdown("---")
        if st.button(
            "Recluster",
            type="secondary",
            key=f"btn_recluster_{self.level}",
        ):
            try:
                self.engine.semantic_recluster(self.level)
                st.success("Reclustered.")
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Recluster failed: {e}")

        st.markdown("---")
        rename_map = self._render_rename_section(layer_df)
        st.markdown("---")
        reassignment_map = self._render_reassignment_section(layer_df)
        st.markdown("---")
        self._render_merge_section(layer_df)
        st.markdown("---")
        self._render_split_section(layer_df)
        st.markdown("---")

        if st.button(
            "Apply Semantic Layer Changes",
            type="primary",
            key=f"btn_apply_all_changes_{self.level}",
        ):
            try:
                self.engine.apply_semantic_changes(
                    level=self.level,
                    rename_map=rename_map,
                    reassignment_map=reassignment_map,
                )
                st.success("Changes applied.")
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Failed to update layer: {e}")
