import streamlit as st

from app.components.editor_attribute_layer import AttributeLayerEditor


# ============================================================
#        Page 4 — Attribute Layer (Step 4 of Wizard)
# ============================================================

class AttributeLayerPage:
    """
    Step 4 — Attribute Layer

    Responsibilities:
    - Display and edit attribute-based clusters
    - Provide clear interface for adjusting attribute-grouped products
    - Prepare data for Step 5 (Hierarchy Browser)

    The engine handles:
    - Attribute sparsity clustering
    - Attribute naming logic
    - Layer refinement
    """

    def __init__(self, wizard):
        self.wizard = wizard

    # --------------------------------------------------------
    # Helpers
    # --------------------------------------------------------

    def _validate_engine(self):
        """Ensure backend exists before editing attribute layer."""
        engine = st.session_state.engine
        if engine is None:
            st.error("Hierarchy engine not initialized. Return to Step 2.")
            if st.button("← Back to Configure"):
                self.wizard.go_to("configure")
            return None
        return engine

    # --------------------------------------------------------
    # Render
    # --------------------------------------------------------

    def render(self):
        st.title("🔬 Step 4 — Attribute Layer")

        engine = self._validate_engine()
        if engine is None:
            return

        st.markdown(
            """
            The **Attribute Layer** groups rows based on their attributes.
            
            You can cluster attributes using either:
            - **Sparsity-based**: cluster by which attributes are present (best for very wide, sparse tables).
            - **Value-based**: cluster by the actual values in attribute columns (best when attributes are more densely filled).
            """
        )

        st.markdown("---")

        # ----------------------------------------------------
        # Method selector
        # ----------------------------------------------------

        current_method = getattr(engine, "get_attribute_method", lambda: "sparsity")()
        config = st.session_state.get("config", {})
        prev_method = config.get("attribute_method", current_method)

        method_options = ["sparsity", "value"]

        def _fmt(m: str) -> str:
            return "Sparsity-based" if m == "sparsity" else "Value-based"

        attr_method = st.radio(
            "Attribute clustering method:",
            options=method_options,
            index=method_options.index(prev_method)
            if prev_method in method_options
            else method_options.index(current_method),
            format_func=_fmt,
            key="attr_method_radio",
        )

        # Persist choice
        st.session_state.config["attribute_method"] = attr_method

        st.markdown("---")

        # ----------------------------------------------------
        # Editor
        # ----------------------------------------------------

        st.subheader("Attribute Layer Editor")

        with st.expander("Edit Attribute Layer", expanded=True):
            editor = AttributeLayerEditor(
                wizard=self.wizard,
                engine=engine,
                title="Attribute Layer Editor",
                method=attr_method,
            )
            editor.render()

        st.markdown("---")

        # ----------------------------------------------------
        # Navigation
        # ----------------------------------------------------

        col1, col2 = st.columns([1, 1])

        with col1:
            st.button("← Back", on_click=self.wizard.prev_page)

        with col2:
            st.button(
                "Next → Explore Hierarchy",
                on_click=self.wizard.next_page,
                type="primary",
            )

