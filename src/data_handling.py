import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from .config import PROCESSED_DIR, DATASETS, RANDOM_STATE


def load_secom():
    """Load UCI SECOM as primary dataset."""
    data = pd.read_csv(DATASETS["secom"], delim_whitespace=True, header=None)
    data.columns = [f"feature_{i}" for i in range(1, data.shape[1] + 1)]

    labels = pd.read_csv(
        DATASETS["secom"].replace(".data", ".labels"),
        delim_whitespace=True,
        header=None,
    )
    labels.columns = ["label", "timestamp"]
    labels["timestamp"] = pd.to_datetime(
        labels["timestamp"], format="%d/%m/%Y %H:%M:%S"
    )
    labels["label"] = labels["label"].map({-1: 0, 1: 1})

    df = pd.concat([data, labels], axis=1)
    df = df.sort_values("timestamp")
    df["lot_id"] = df["timestamp"].dt.date
    return df


def preprocess_df(df, save_path=None):
    """Impute, scale, drop low-var features."""
    imputer = SimpleImputer(strategy="median")
    df.iloc[:, :-3] = imputer.fit_transform(
        df.iloc[:, :-3]
    )  # Exclude label, timestamp, lot_id

    scaler = StandardScaler()
    df.iloc[:, :-3] = scaler.fit_transform(df.iloc[:, :-3])

    low_var_cols = [col for col in df.columns[:-3] if df[col].var() < 0.01]
    df.drop(low_var_cols, axis=1, inplace=True)

    if save_path:
        df.to_csv(save_path, index=False)
    return df


# Similar functions for secondary datasets (stubbed for brevity; expand as needed)
def load_secondary(dataset_name):
    if dataset_name == "steel_plates":
        df = pd.read_csv(DATASETS["steel_plates"])
        # Binary encode faults, etc.
        return df
    # Add for APS, CMAPSS...
    raise ValueError(f"Dataset {dataset_name} not implemented.")
