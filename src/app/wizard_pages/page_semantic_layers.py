import streamlit as st

from app.components.editor_semantic_layer import SemanticLayerEditor


# ============================================================
#    Page 3 — Semantic Layers (Level 0 & Level 1) — Step 3
# ============================================================

class SemanticLayersPage:
    """
    Step 3 — Edit Semantic Layers 1 and 0
    """

    def __init__(self, wizard):
        self.wizard = wizard

    def render(self):
        st.title("🧠 Step 3 — Semantic Layers")

        engine = st.session_state.engine
        if engine is None:
            st.error("Hierarchy engine not initialized.")
            return

        st.markdown("""
        Adjust the semantic layers using rename, move, merge, and split operations.
        """)

        st.markdown("---")
        st.subheader("Semantic Layer 1")
        with st.expander("Edit Semantic Layer 1", expanded=True):
            SemanticLayerEditor(1, self.wizard).render()

        st.markdown("---")
        st.subheader("Semantic Layer 0")
        with st.expander("Edit Semantic Layer 0", expanded=False):
            SemanticLayerEditor(0, self.wizard).render()

        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.button("← Back", on_click=self.wizard.prev_page)
        with col2:
            st.button("Next → Attribute Layer", on_click=self.wizard.next_page, type="primary")
