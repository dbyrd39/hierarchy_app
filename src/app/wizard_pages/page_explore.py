import streamlit as st

from app.components.hierarchy_browser import HierarchyBrowser


# ============================================================
#        Page 5 — Explore Hierarchy (Step 5 of Wizard)
# ============================================================

class ExplorePage:
    """
    Step 5 — Explore Hierarchy

    Responsibilities:
    - Show final combined hierarchy (Semantic Layer 0 → Semantic Layer 1 → Categories → Attribute Layer)
    - Provide interactive tree/file-system view
    - Allow visual inspection before exporting
    - Ensures hierarchy_df is present
    """

    def __init__(self, wizard):
        self.wizard = wizard

    # --------------------------------------------------------
    # Helpers
    # --------------------------------------------------------

    def _validate_hierarchy(self):
        """Ensure the final hierarchy exists."""
        engine = st.session_state.engine
        if engine is None:
            st.error("Hierarchy engine not initialized. Return to Step 2.")
            if st.button("← Back to Configure"):
                self.wizard.go_to("configure")
            return None

        df = engine.get_hierarchy_df()
        if df is None or df.empty:
            st.error("Hierarchy has not been generated yet.")
            if st.button("← Back to Semantic Layers"):
                self.wizard.go_to("semantic_layers")
            return None

        return df

    # --------------------------------------------------------
    # Render
    # --------------------------------------------------------

    def render(self):
        st.title("📂 Step 5 — Explore Hierarchy")

        df_h = self._validate_hierarchy()
        if df_h is None:
            return

        st.markdown(
            """
            The hierarchy below displays the complete structure built by the BL3 engine:
            
            - **Semantic Layer 0** (top-level semantic clusters)  
            - **Semantic Layer 1**  
            - **Categories**  
            - **Attribute Layer** (attribute-based clusters)
            """
        )

        st.markdown("---")

        # ----------------------------------------------------
        # Browser Component
        # ----------------------------------------------------

        browser = HierarchyBrowser(df_hierarchy=df_h)
        browser.render()

        st.markdown("---")

        # ----------------------------------------------------
        # Navigation
        # ----------------------------------------------------

        col1, col2 = st.columns([1, 1])

        with col1:
            st.button("← Back", on_click=self.wizard.prev_page)

        with col2:
            st.button(
                "Next → Download",
                on_click=self.wizard.next_page,
                type="primary",
            )
