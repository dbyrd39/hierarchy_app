# analysis/__init__.py

"""
Analysis utilities for exploring attribute coverage, category/subcluster patterns,
and derived statistics from the enriched product hierarchy dataset.

This package exposes only the reusable analysis functions.
To run the full standalone analysis script, execute:

    python src/analysis/subcluster_analysis.py
"""

from .subcluster_analysis import (
    attribute_coverage_by_cluster,
    attribute_coverage_by_category_and_subcluster,
    get_attribute_columns,
)

__all__ = [
    "attribute_coverage_by_cluster",
    "attribute_coverage_by_category_and_subcluster",
    "get_attribute_columns",
]
