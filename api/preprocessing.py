import pandas as pd


def clean_prediction_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and format raw patient diagnostic records before heart disease inference.
    """
    df = df.copy()

    # Drop non-predictive identifiers or target leakage if present
    drop_cols = [c for c in ["id", "customerID", "dataset", "num", "target", "Churn"] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    # Standardize boolean or string flags
    if "fbs" in df.columns:
        df["fbs"] = df["fbs"].apply(
            lambda v: True if str(v).strip().lower() in ["true", "1", "yes"] else False
        )
    if "exang" in df.columns:
        df["exang"] = df["exang"].apply(
            lambda v: True if str(v).strip().lower() in ["true", "1", "yes"] else False
        )

    # Ensure numeric columns are float/int
    num_cols = ["age", "trestbps", "chol", "thalch", "oldpeak", "ca"]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df
