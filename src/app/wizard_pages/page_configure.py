import streamlit as st
import pandas as pd
from typing import Optional

from core.attribute_layer import _select_attribute_columns
from core.category_layer import ensure_or_generate_category_name




# ============================================================
#            Page 2 — Configure Hierarchy (Step 2)
# ============================================================

class ConfigurePage:
    """
    Step 2 — Configure Hierarchy

    Responsibilities:
    - Select the category column (or auto-detect)
    - Display column-level summary
    - Initialize the HierarchyEngine once category_col is chosen
    - Allow highly polished, guided flow to the Semantic Layer step

    This page forms the bridge from raw data → backend hierarchy engine.
    """

    def __init__(self, wizard):
        self.wizard = wizard

    # --------------------------------------------------------
    # Category auto-detection
    # --------------------------------------------------------

    def _auto_detect_category_column(self, df: pd.DataFrame) -> Optional[str]:
        """
        Heuristic: find columns that look like product category text.
        Backend will support a more advanced autodetection engine,
        but this minimal version works reliably.
        """
        text_like_columns = []
        for col in df.columns:
            # must be object or string dtype
            if df[col].dtype == object:
                sample = df[col].dropna().astype(str).head(50)
                avg_len = sample.map(len).mean() if not sample.empty else 0
                # heuristic: categories typically ~5–50 chars
                if 3 < avg_len < 60:
                    text_like_columns.append(col)

        if not text_like_columns:
            return None

        # Favor columns with "category" in name
        for col in text_like_columns:
            if "category" in col.lower():
                return col

        # fallback: first candidate
        return text_like_columns[0]

    # --------------------------------------------------------
    # Render
    # --------------------------------------------------------

    def render(self):
        st.title("⚙️ Step 2 — Configure Hierarchy")

        # ----------------------------------------------------
        # Validate data availability
        # ----------------------------------------------------
        df = st.session_state.get("clean_df", None)
        if df is None:
            st.error("No dataset found. Please return to Step 1 and upload a file.")
            if st.button("← Back to Upload"):
                self.wizard.go_to("upload")
            return

        st.markdown(
            """
            Select the column that represents **categories**.
            The hierarchy engine uses this column as the base level from
            which semantic and attribute layers are built.
            """
        )

        st.markdown("---")

        # ----------------------------------------------------
        # Column selection UI
        # ----------------------------------------------------

        columns = list(df.columns)

        # Auto-detection
        autodetected = self._auto_detect_category_column(df)

        st.subheader("Choose Category Column")

        CATEGORY_AUTO = "(auto-detect)"
        CATEGORY_SYNTHETIC = "(auto-generate synthetic categories)"

        category_choice = st.selectbox(
            "Select the category name column:",
            options=[CATEGORY_AUTO, CATEGORY_SYNTHETIC] + columns,
            index=0 if autodetected else 2,  # default to auto-detect if possible
            help=(
                "Choose an existing text column for categories, or let the app "
                "auto-generate synthetic categories based on attribute patterns."
            ),
        )

        # Resolve choice → category_col (may be None if synthetic)
        if category_choice == CATEGORY_SYNTHETIC:
            category_col = None
        elif category_choice == CATEGORY_AUTO:
            category_col = autodetected
        else:
            category_col = category_choice

        # Display auto-detection results / warnings
        if category_choice == CATEGORY_SYNTHETIC:
            st.info(
                "You chose to **auto-generate synthetic categories**. "
                "The app will group rows using attribute sparsity patterns "
                "and use those group names as `category_name`."
            )
        else:
            if autodetected:
                st.info(f"Auto-detected category column: **{autodetected}**")
            else:
                st.warning(
                    "Auto-detection failed. Please select the correct category column "
                    "or choose to auto-generate synthetic categories."
                )

            if category_col is None:
                st.error("Please choose a valid category column or the synthetic option.")
                return

        st.markdown("---")

        # ----------------------------------------------------
        # Preview selected or synthetic category
        # ----------------------------------------------------

        if category_col is not None:
            # Simple preview of the chosen existing column
            st.subheader(f"Preview — `{category_col}`")
            st.dataframe(df[[category_col]].head(20), use_container_width=True)
        else:
            st.subheader("Preview — synthetic categories")

            # For preview, optionally sample to avoid huge runtime on massive tables
            if len(df) > 2000:
                df_preview = df.sample(2000, random_state=42).reset_index(drop=True)
            else:
                df_preview = df.copy().reset_index(drop=True)

            try:
                # Use the SAME logic the engine uses to generate category_name
                df_synth = ensure_or_generate_category_name(
                    df_preview,
                    category_col=None,
                    extra_excluded_cols=None,  # preview without exclusions
                )

                st.dataframe(
                    df_synth[["category_name"]].head(20),
                    use_container_width=True,
                )
                st.caption(
                    "These synthetic categories are generated from attribute sparsity "
                    "patterns. Final categories may change slightly if you adjust "
                    "attribute exclusions below."
                )
            except Exception as e:
                st.warning(
                    f"Could not preview synthetic categories (they will still be "
                    f"generated when initializing the engine). Error: {e}"
                )

        st.markdown("---")


        # ----------------------------------------------------
        # Attribute clustering: optional excluded columns
        # ----------------------------------------------------

        st.subheader("Attribute Clustering — Exclude Columns (Optional)")

        # Previous selection (if any)
        prev_excluded = st.session_state.get("config", {}).get(
            "attribute_excluded_cols", []
        )

        candidate_cols = [
            c for c in df.columns
            if category_col is None or c != category_col
        ]

        excluded_cols = st.multiselect(
            "Columns to exclude from attribute clustering:",
            options=candidate_cols,
            default=prev_excluded,
            help=(
                "These columns will NOT be used when forming attribute clusters "
                "within categories. Useful for metadata or noisy fields."
            ),
        )

        # Persist in session config
        st.session_state.config.setdefault("attribute_excluded_cols", [])
        st.session_state.config["attribute_excluded_cols"] = excluded_cols

        st.markdown("---")

        # ----------------------------------------------------
        # Attribute clustering method: sparsity vs value
        # ----------------------------------------------------

        st.subheader("Attribute Clustering — Method")

        # Compute a simple sparsity heuristic on attribute columns
        try:
            attr_cols = _select_attribute_columns(df, extra_excluded_cols=excluded_cols)
        except Exception:
            attr_cols = []

        recommended_method = "sparsity"
        heuristic_msg = "Unable to compute attribute sparsity; defaulting to sparsity-based clustering."

        if attr_cols:
            sub_df = df[attr_cols]
            non_null_frac = 1.0 - sub_df.isna().mean().mean()
            # Heuristic: very sparse → sparsity; otherwise → value
            if non_null_frac < 0.2:
                recommended_method = "sparsity"
                heuristic_msg = (
                    f"Detected high sparsity in attribute columns "
                    f"(~{non_null_frac:.1%} non-null). "
                    f"Recommended: **sparsity-based** attribute clustering."
                )
            else:
                recommended_method = "value"
                heuristic_msg = (
                    f"Detected moderate/low sparsity in attribute columns "
                    f"(~{non_null_frac:.1%} non-null). "
                    f"Recommended: **value-based** attribute clustering."
                )

        st.caption(heuristic_msg)

        prev_method = st.session_state.get("config", {}).get(
            "attribute_method",
            recommended_method,
        )

        method_options = ["sparsity", "value"]

        def _fmt_method(m: str) -> str:
            label = "Sparsity-based" if m == "sparsity" else "Value-based"
            if m == recommended_method:
                return f"{label} (recommended)"
            return label

        # Choose method (user can override recommendation)
        attr_method = st.radio(
            "Choose attribute clustering method:",
            options=method_options,
            index=method_options.index(prev_method)
            if prev_method in method_options
            else method_options.index(recommended_method),
            format_func=_fmt_method,
        )

        st.session_state.config["attribute_method"] = attr_method

        st.markdown("---")

        # ----------------------------------------------------
        # Initialize Hierarchy Engine
        # ----------------------------------------------------

        if st.button("Initialize Hierarchy Engine", type="primary"):
            self.wizard.init_engine(
                df,
                category_col,
                attribute_excluded_cols=excluded_cols,
                attribute_method=attr_method,
            )
            st.success("Hierarchy engine initialized successfully!")
            st.session_state.category_col = category_col

        # ----------------------------------------------------
        # Only allow Next if engine is initialized
        # ----------------------------------------------------

        engine_ready = st.session_state.engine is not None

        st.markdown("---")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.button("← Back", on_click=self.wizard.prev_page)

        with col2:
            st.button(
                "Next → Semantic Layers",
                on_click=self.wizard.next_page,
                type="primary",
                disabled=not engine_ready,
            )

