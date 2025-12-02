# analysis/subcluster_analysis.py

import pandas as pd
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# Utility: determine which columns to use as attribute columns
# ------------------------------------------------------------
def get_attribute_columns(df: pd.DataFrame) -> list[str]:
    """
    Determines which columns of the clustered dataset should be treated
    as attribute columns for coverage analysis. Excludes:

    - First 26 metadata columns
    - Last 4 administrative columns
    - Any column ending in '_unit'
    """
    cols_first_26 = list(df.columns[:26])
    cols_last_4 = list(df.columns[-4:])
    positional_excluded = set(cols_first_26 + cols_last_4)

    unit_cols = {c for c in df.columns if c.lower().endswith("_unit")}

    excluded = positional_excluded | unit_cols
    return [c for c in df.columns if c not in excluded]


# ------------------------------------------------------------
# Analysis functions (safe to import via analysis.__init__)
# ------------------------------------------------------------
def attribute_coverage_by_cluster(
    df: pd.DataFrame, cluster_col: str, attribute_cols: list[str]
) -> pd.DataFrame:
    """
    Returns a DataFrame where:
        index = cluster IDs
        columns = attribute names
        values = % non-null values within each cluster
    """
    non_null = df[attribute_cols].notna().astype(int)
    coverage = non_null.groupby(df[cluster_col]).mean() * 100.0
    coverage.index.name = cluster_col
    return coverage.round(1)


def attribute_coverage_by_category_and_subcluster(
    df: pd.DataFrame, attribute_cols: list[str]
) -> pd.DataFrame:
    """
    Returns:
        index = (category_name, category_subcluster)
        columns = attributes
        values = % non-null values within that (category, subcluster)
    """
    non_null = df[attribute_cols].notna().astype(int)

    coverage = (
        non_null
        .groupby([df["category_name"], df["category_subcluster"]])
        .mean()
        * 100.0
    )

    coverage.index.set_names(["category_name", "category_subcluster"], inplace=True)
    return coverage.round(1)


# ------------------------------------------------------------
# Standalone Analysis Script
# ONLY runs when executed directly:
#     python src/analysis/subcluster_analysis.py
# ------------------------------------------------------------
if __name__ == "__main__":
    print("Loading office_products_enriched.csv ...")

    try:
        cluster_df = pd.read_csv("office_products_enriched.csv", low_memory=False)
    except FileNotFoundError:
        print("\n❌ ERROR: 'office_products_enriched.csv' not found.\n")
        print("Run the Streamlit app first and export the enriched CSV.")
        exit(1)

    print("✔ Loaded dataset:", cluster_df.shape)

    # Determine attribute columns
    attribute_cols = get_attribute_columns(cluster_df)
    print(f"✔ Detected {len(attribute_cols)} attribute columns.")

    # --------------------------------------------------------
    # Global coverage by category_subcluster
    # --------------------------------------------------------
    cat_cov = attribute_coverage_by_cluster(
        cluster_df, "category_subcluster", attribute_cols
    )

    # Top 20 strongest attributes
    top_attrs = cat_cov.mean().sort_values(ascending=False).head(20).index
    cat_cov_top = cat_cov[top_attrs]

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(cat_cov_top, aspect="auto")
    fig.colorbar(im, ax=ax, label="% non-null")

    ax.set_xticks(range(len(cat_cov_top.columns)))
    ax.set_xticklabels(cat_cov_top.columns, rotation=90)
    ax.set_xlabel("Product Attribute")

    ax.set_yticks(range(len(cat_cov_top.index)))
    ax.set_yticklabels(cat_cov_top.index)
    ax.set_ylabel("Category Subcluster")

    fig.suptitle("Attribute Coverage by Category Subcluster (Top 20 Attributes)", fontsize=14)

    fig.text(
        0.5,
        -0.05,
        "Each cell represents the % of products in a subcluster that have "
        "a non-null value for the specified attribute.",
        ha="center",
        va="top",
        wrap=True,
    )

    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------
    # Detailed view for a single (category, subcluster)
    # --------------------------------------------------------
    cat_cov_by_pair = attribute_coverage_by_category_and_subcluster(
        cluster_df, attribute_cols
    )

    target_category = "Board Accessories"
    target_subcluster = 0

    key = (target_category, target_subcluster)

    if key not in cat_cov_by_pair.index:
        print(f"No rows found for category='{target_category}', subcluster={target_subcluster}")
    else:
        row = (
            cat_cov_by_pair.loc[key]
            .sort_values(ascending=False)
            .head(15)
        )

        print(
            f"✔ Top attributes for category='{target_category}', "
            f"subcluster={target_subcluster} (n={len(row)})"
        )

        plt.figure(figsize=(8, 4))
        plt.bar(row.index, row.values)
        plt.xticks(rotation=90)
        plt.ylabel("% non-null")
        plt.title(
            f"Top Attributes – category='{target_category}', "
            f"subcluster={target_subcluster}"
        )
        plt.tight_layout()
        plt.show()

    print("\n✔ Analysis complete.")

