import pandas as pd
from scipy.stats import boxcox
from .config import PROCESSED_DIR


def engineer_lots(df):
    """Aggregate to lots for count models."""
    lot_df = (
        df.groupby("lot_id")
        .agg(
            {
                "label": "sum",  # Defect count
                **{col: "mean" for col in df.columns if col.startswith("feature_")},
            }
        )
        .rename(columns={"label": "defect_count"})
    )
    lot_df["defect_count"], _ = boxcox(lot_df["defect_count"] + 1)
    lot_df.to_csv(os.path.join(PROCESSED_DIR, "lot_data.csv"), index=True)
    return lot_df


# Add feature selection (e.g., correlation > 0.5)
def select_features(df, threshold=0.5):
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return df.drop(to_drop, axis=1)
