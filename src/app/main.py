import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


import streamlit as st
import pandas as pd
import os
from typing import Dict, Optional, Any

# Backend engine
from core.hierarchy_engine import HierarchyEngine

# Wizard pages
from app.wizard_pages.page_upload import UploadPage
from app.wizard_pages.page_configure import ConfigurePage
from app.wizard_pages.page_semantic_layers import SemanticLayersPage
from app.wizard_pages.page_attribute_layer import AttributeLayerPage
from app.wizard_pages.page_explore import ExplorePage
from app.wizard_pages.page_download import DownloadPage


# ============================================================
#                      Wizard Controller
# ============================================================

class HierarchyWizard:
    """
    HierarchyWizard orchestrates the entire multi-step hierarchy-building
    workflow. It manages navigation between wizard steps, holds global
    session state, and provides helper utilities.

    The UI structure is:

        Step 1 — Upload Data
        Step 2 — Configure Hierarchy
        Step 3 — Semantic Layers (Level 0 & Level 1)
        Step 4 — Attribute Layer
        Step 5 — Explore Hierarchy
        Step 6 — Download

    Navigation is handled through explicit page registration.
    """

    def __init__(self):
        self._init_session_state()
        self._register_pages()

    # --------------------------------------------------------
    # Initialization
    # --------------------------------------------------------

    def _init_session_state(self):
        """Initializes Streamlit session state variables used across steps."""
        defaults = {
            "page": "upload",             # current wizard step
            "raw_df": None,               # uploaded raw dataset
            "clean_df": None,             # cleaned + standardized dataframe
            "category_col": None,         # selected or auto-detected category column
            "engine": None,               # instance of HierarchyEngine (backend)
            "hierarchy_df": None,         # full hierarchical output
            "config": {},                 # dynamic config options
            "semantic_info": {},          # results of semantic layer editing
            "attribute_info": {},         # results of attribute layer editing
            "ready_for_export": False,    # controls final download step
        }

        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

    def _register_pages(self):
        """Explicitly registers wizard step pages."""
        self.pages: Dict[str, Any] = {
            "upload": UploadPage(self),
            "configure": ConfigurePage(self),
            "semantic_layers": SemanticLayersPage(self),
            "attribute_layer": AttributeLayerPage(self),
            "explore": ExplorePage(self),
            "download": DownloadPage(self),
        }

    # --------------------------------------------------------
    # Navigation logic (Next / Back)
    # --------------------------------------------------------

    def go_to(self, page_name: str):
        """Direct navigation to a page."""
        if page_name in self.pages:
            st.session_state.page = page_name

    def next_page(self):
        """Navigate to the next step in the wizard (guided workflow)."""
        order = list(self.pages.keys())
        current = st.session_state.page
        try:
            idx = order.index(current)
            next_page = order[idx + 1]
            st.session_state.page = next_page
        except (ValueError, IndexError):
            pass  # Already at last page

    def prev_page(self):
        """Navigate to the previous step."""
        order = list(self.pages.keys())
        current = st.session_state.page
        try:
            idx = order.index(current)
            prev_page = order[idx - 1]
            st.session_state.page = prev_page
        except (ValueError, IndexError):
            pass

    # --------------------------------------------------------
    # Backend engine
    # --------------------------------------------------------

    def init_engine(
        self,
        df: pd.DataFrame,
        category_col: Optional[str],
        attribute_excluded_cols: list[str] | None = None,
        attribute_method: str = "sparsity",
    ):
        """
        Creates a fresh HierarchyEngine instance using the redesigned backend.
        This engine performs all semantic and attribute-level clustering.

        - `category_col` may be None (auto-generate categories).
        - `attribute_excluded_cols` controls which columns are ignored during
          attribute-layer clustering and naming.
        - `attribute_method` selects 'sparsity' or 'value' clustering.
        """
        engine = HierarchyEngine(
            df=df,
            category_col=category_col,
            attribute_method=attribute_method,
            attribute_excluded_cols=attribute_excluded_cols,
        )

        st.session_state.engine = engine


    # --------------------------------------------------------
    # Main rendering pipeline
    # --------------------------------------------------------

    def render_sidebar(self):
        """Renders the left navigation panel."""
        st.sidebar.title("📊 Product Hierarchy Wizard")

        menu_items = {
            "upload": "Step 1 — Upload Data",
            "configure": "Step 2 — Configure Hierarchy",
            "semantic_layers": "Step 3 — Semantic Layers (0 & 1)",
            "attribute_layer": "Step 4 — Attribute Layer",
            "explore": "Step 5 — Explore Hierarchy",
            "download": "Step 6 — Download",
        }

        st.sidebar.markdown("---")

        for key, label in menu_items.items():
            if st.sidebar.button(label, key=f"nav_{key}"):
                self.go_to(key)

        st.sidebar.markdown("---")
        st.sidebar.caption("Use the buttons above to move across workflow steps.")

    def render_page(self):
        """Dispatches to the appropriate page based on session_state.page."""
        page_name = st.session_state.page
        page = self.pages.get(page_name)

        if page is None:
            st.error(f"Unknown page: {page_name}")
            return

        page.render()

    # --------------------------------------------------------
    # Public entry point
    # --------------------------------------------------------

    def run(self):
        """Starts the wizard application."""
        self.render_sidebar()
        self.render_page()


# ============================================================
#                       Streamlit Entry
# ============================================================

def run_app():
    st.set_page_config(
        page_title="Hierarchy Builder",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    wizard = HierarchyWizard()
    wizard.run()


if __name__ == "__main__":
    run_app()
