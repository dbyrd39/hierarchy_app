import streamlit as st
import pandas as pd


from preprocessing import clean_office_products


# ============================================================
#                   Page 1 — Upload Data
# ============================================================

class UploadPage:
    """
    Step 1 of the wizard: Upload & validate the dataset.

    Responsibilities:
    - File upload (CSV/Parquet)
    - Basic file inspection
    - Validation (non-empty, tabular)
    - Pushes cleaned df into wizard.session_state["clean_df"]
    - Enables progression to Step 2

    This page does not perform semantic or attribute layer logic.
    It simply prepares the raw data for the backend engine.
    """

    def __init__(self, wizard):
        self.wizard = wizard

    # --------------------------------------------------------
    # Utilities
    # --------------------------------------------------------

    def _load_csv(self, uploaded_file) -> pd.DataFrame:
        """Load CSV using pandas with fallback robustness."""
        try:
            return pd.read_csv(uploaded_file, low_memory=False)
        except UnicodeDecodeError:
            # Handle common encoding issues
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, low_memory=False, encoding="latin-1")

    def _load_parquet(self, uploaded_file) -> pd.DataFrame:
        """Load Parquet file."""
        return pd.read_parquet(uploaded_file)

    # --------------------------------------------------------
    # Main render
    # --------------------------------------------------------

    def render(self):
        st.title("📁 Step 1 — Upload Your Dataset")
        st.markdown(
            """
            Upload a **CSV** or **Parquet** file containing your master data.
            The file must include at least one column with category name text.
            """
        )

        st.markdown("---")

        uploaded_file = st.file_uploader(
            "Choose a file (.csv or .parquet)", type=["csv", "parquet"]
        )

        if uploaded_file is not None:
            df = None

            # ------------------------------------------------
            # Load according to file type
            # ------------------------------------------------
            if uploaded_file.name.lower().endswith(".csv"):
                df = self._load_csv(uploaded_file)
            elif uploaded_file.name.lower().endswith(".parquet"):
                df = self._load_parquet(uploaded_file)

            # ------------------------------------------------
            # Validate
            # ------------------------------------------------
            if df is None or df.empty:
                st.error("The uploaded file is empty or unreadable.")
                return

            st.success("File uploaded successfully!")

            st.markdown("### Preview of Uploaded Data")
            st.dataframe(df.head(20), use_container_width=True)

            st.markdown("---")

            # Save to session state
            st.session_state.raw_df = df

            # Run numeric + unit preprocessing for the "clean" version
            try:
                df_clean = clean_office_products(df)
            except Exception as e:
                st.warning(
                    f"Preprocessing (numeric units) failed; using raw data as clean_df. Error: {e}"
                )
                df_clean = df.copy()

            st.session_state.clean_df = df_clean


            # Enable navigation only after successful upload
            st.button(
                "Next → Configure Hierarchy",
                on_click=self.wizard.next_page,
                type="primary",
            )

        else:
            st.info("Please upload a file to proceed.")
