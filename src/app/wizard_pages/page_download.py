import streamlit as st
import io


# ============================================================
#          Page 6 — Download Final Hierarchy (Step 6)
# ============================================================

class DownloadPage:
    """
    Step 6 — Download

    Responsibilities:
    - Display final hierarchy summary
    - Export final hierarchy dataframe to CSV / Parquet
    - Allow optional JSON export for downstream integrations
    - Final step of the wizard workflow
    """

    def __init__(self, wizard):
        self.wizard = wizard

    # --------------------------------------------------------
    # Validate Engine & Hierarchy
    # --------------------------------------------------------

    def _validate_hierarchy(self):
        engine = st.session_state.engine
        if engine is None:
            st.error("Hierarchy engine is not initialized.")
            if st.button("← Back to Configure"):
                self.wizard.go_to("configure")
            return None

        df_h = engine.get_hierarchy_df()
        if df_h is None or df_h.empty:
            st.error("Hierarchy not generated yet.")
            if st.button("← Back to Semantic Layers"):
                self.wizard.go_to("semantic_layers")
            return None

        return df_h

    # --------------------------------------------------------
    # Render
    # --------------------------------------------------------

    def render(self):
        st.title("📥 Step 6 — Download Final Hierarchy")

        df_h = self._validate_hierarchy()
        if df_h is None:
            return

        st.markdown(
            """
            Your final hierarchy is ready for export.  
            You may download it as **CSV**, **Parquet**, or **JSON structure**  
            for use in your downstream systems or analytics workflows.
            """
        )

        st.markdown("---")

        # ----------------------------------------------------
        # Data Preview
        # ----------------------------------------------------

        st.subheader("Preview of Final Hierarchy")
        st.dataframe(df_h.head(50), use_container_width=True)

        st.markdown("---")

        # ----------------------------------------------------
        # CSV Download
        # ----------------------------------------------------

        csv_buffer = io.StringIO()
        df_h.to_csv(csv_buffer, index=False)
        st.download_button(
            label="⬇️ Download as CSV",
            data=csv_buffer.getvalue(),
            file_name="hierarchy.csv",
            mime="text/csv",
        )

        # ----------------------------------------------------
        # Parquet Download
        # ----------------------------------------------------

        parquet_buffer = io.BytesIO()
        df_h.to_parquet(parquet_buffer, index=False)
        st.download_button(
            label="⬇️ Download as Parquet",
            data=parquet_buffer.getvalue(),
            file_name="hierarchy.parquet",
            mime="application/octet-stream",
        )

        # ----------------------------------------------------
        # JSON Export (Optional)
        # ----------------------------------------------------

        json_data = df_h.to_json(orient="records")
        st.download_button(
            label="⬇️ Download as JSON",
            data=json_data,
            file_name="hierarchy.json",
            mime="application/json",
        )

        st.markdown("---")

        # ----------------------------------------------------
        # Navigation
        # ----------------------------------------------------

        col1, col2 = st.columns([1, 1])

        with col1:
            st.button("← Back", on_click=self.wizard.prev_page)

        with col2:
            st.button(
                "Finish",
                on_click=lambda: self.wizard.go_to("upload"),
                type="primary",
            )

        st.caption("Click 'Finish' to restart the wizard from Step 1.")
