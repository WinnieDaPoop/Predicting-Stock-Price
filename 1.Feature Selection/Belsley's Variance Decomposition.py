"""Belsley's variance decomposition analysis for the stock training set.

This version follows the SVD-based method in your example:
- optionally scales columns to unit length
- uses singular values to compute condition indices
- computes variance decomposition proportions from V and singular values

Outputs written next to the script:
- belsley_condition_indices.csv
- belsley_variance_decomposition.csv
- belsley_collinearity_flags.csv
"""

from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_PATH = PROJECT_DIR / "train.xlsx"

TARGET_COLUMN = "target"
ID_COLUMNS = {"id", "stock_id", TARGET_COLUMN}

# Typical Belsley thresholds used to flag near-dependencies.
CONDITION_INDEX_THRESHOLD = 30.0
VARIANCE_PROPORTION_THRESHOLD = 0.5
MIN_HIGH_CONDITION_COMPONENTS = 2

# Match the behavior of your example.
SCALE_COLUMNS = False


def load_training_data(path: Path) -> pd.DataFrame:
    """Load the first sheet from the training workbook.

    This is the starting point of the analysis. The script reads the Excel file
    and brings the data into a pandas DataFrame so we can filter the columns.
    """
    if not path.exists():
        raise FileNotFoundError(f"Training workbook not found: {path}")

    return pd.read_excel(path, sheet_name=0)


def select_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Select numeric predictors and drop identifier/target columns.

    This section keeps only the columns that should be used as input features.
    It removes `id`, `stock_id`, and `target`, then handles missing or constant
    columns so the decomposition can run safely.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in ID_COLUMNS]

    if not feature_cols:
        raise ValueError("No numeric feature columns found after exclusions.")

    X = df[feature_cols].copy()

    # Remove columns that contain only missing values.
    X = X.dropna(axis=1, how="all")

    # Remove columns that do not vary, because they cannot contribute to the analysis.
    constant_cols = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
    if constant_cols:
        X = X.drop(columns=constant_cols)
        warnings.warn(
            f"Dropped constant columns: {constant_cols}",
            RuntimeWarning,
            stacklevel=2,
        )

    # Fill remaining missing values with the median so the matrix stays numeric.
    if X.isna().any().any():
        X = X.fillna(X.median(numeric_only=True))

    if X.shape[1] < 2:
        raise ValueError("Need at least two numeric features for Belsley analysis.")

    return X


def build_design_matrix(X: pd.DataFrame, scale_columns: bool = False) -> tuple[np.ndarray, pd.Index]:
    """Convert the features to a NumPy matrix and optionally scale column norms.

    If `scale_columns` is True, each feature is divided by its Euclidean norm.
    This matches the style of some Belsley implementations that normalize the
    matrix before decomposition.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame for feature selection.")

    feature_names = X.columns
    matrix = X.to_numpy(dtype=float)

    if scale_columns:
        col_norms = np.linalg.norm(matrix, axis=0)
        zero_norms = col_norms == 0
        if zero_norms.any():
            raise ValueError("Cannot scale columns with zero norm.")
        matrix = matrix / col_norms

    return matrix, feature_names


def belsley_variance_decomposition(
    X: pd.DataFrame,
    scale_columns: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute condition indices and variance decomposition proportions using SVD.

    This is the core Belsley step. The feature matrix is decomposed into its
    singular vectors and singular values, then those values are used to measure
    how much each component contributes to instability in the regression space.
    """
    X_matrix, feature_names = build_design_matrix(X, scale_columns=scale_columns)

    # Step 1: singular value decomposition.
    # X = U * diag(s) * Vt, where s contains the singular values.
    U, s, Vt = np.linalg.svd(X_matrix, full_matrices=False)
    V = Vt.T

    # Step 2: condition indices.
    # Large condition indices indicate that one singular direction is much smaller
    # than the largest one, which is a warning sign for multicollinearity.
    s_safe = np.where(s <= np.finfo(float).eps, np.finfo(float).eps, s)
    condition_indices = s_safe.max() / s_safe

    # Step 3: variance decomposition proportions.
    # These proportions show which features are heavily associated with each
    # near-degenerate component.
    variance_components = (V ** 2) / (s_safe ** 2) # this to to get the component of variances of feature
    variance_sum_per_variable = variance_components.sum(axis=1, keepdims=True) # this to to get the sum of the variances of feature
    variance_sum_per_variable = np.where(
        variance_sum_per_variable == 0, 
        np.finfo(float).eps,    
        variance_sum_per_variable,
    )
    vdps = (variance_components / variance_sum_per_variable).T # this to to get the variance decomposition proportions

    condition_df = pd.DataFrame(
        {
            "singular_value": s,
            "condition_index": condition_indices,
        },
        index=[f"component_{i+1}" for i in range(len(s))],
    )

    # Put the variance proportions into a table so each row is a component and
    # each column is a feature.
    varprop_df = pd.DataFrame(
        vdps, # this to to get the variance decomposition proportions
        index=condition_df.index, # this to to get the component names
        columns=feature_names, # this to to get the feature names
    )

    # Find components that look suspicious: a high condition index combined with
    # several features having large variance proportions on the same component.
    flags = []
    for component in condition_df.index:
        ci = float(condition_df.loc[component, "condition_index"])
        high_props = varprop_df.loc[component] >= VARIANCE_PROPORTION_THRESHOLD
        flagged_features = varprop_df.columns[high_props].tolist()
        if ci >= CONDITION_INDEX_THRESHOLD and len(flagged_features) >= MIN_HIGH_CONDITION_COMPONENTS:
            flags.append(
                {
                    "component": component,
                    "condition_index": ci,
                    "flagged_features": ", ".join(flagged_features),
                    "num_flagged_features": len(flagged_features),
                }
            )

    flags_df = pd.DataFrame(flags)
    return condition_df, varprop_df, flags_df



def main() -> None:
    # Load the workbook and prepare the feature matrix.
    df = load_training_data(DATA_PATH)
    X = select_feature_matrix(df)

    # Run the Belsley analysis.
    condition_df, varprop_df, flags_df = belsley_variance_decomposition(
        X,
        scale_columns=SCALE_COLUMNS,
    )

    # Save the results into a single Excel workbook with one sheet per table.
    output_path = SCRIPT_DIR / "belsley_analysis.xlsx"
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        condition_df.to_excel(writer, sheet_name="condition_indices")
        varprop_df.to_excel(writer, sheet_name="variance_decomposition")
        flags_df.to_excel(writer, sheet_name="collinearity_flags", index=False)

    # Print a short summary for the console.
    print(f"Loaded training data: {df.shape[0]:,} rows x {df.shape[1]:,} columns")
    print(f"Analyzed features: {X.shape[1]:,}")
    print(f"Scale columns: {SCALE_COLUMNS}")
    print(f"Saved all analysis tables to: {output_path}")

    top_components = condition_df.sort_values("condition_index", ascending=False).head(10)
    print("\nTop condition indices:")
    print(top_components.to_string())

    if flags_df.empty:
        print("\nNo components exceeded the default collinearity thresholds.")
    else:
        print("\nPotential multicollinearity flags:")
        print(flags_df.to_string(index=False))


if __name__ == "__main__":
    main()
