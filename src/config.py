# Configuration for Automotive Defect Prediction Project
import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
EXTERNAL_DIR = os.path.join(DATA_DIR, "external")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
FIGURES_DIR = os.path.join(BASE_DIR, "reports", "figures")

# Dataset params
DATASETS = {
    "secom": os.path.join(EXTERNAL_DIR, "secom/secom.data"),
    "steel_plates": os.path.join(EXTERNAL_DIR, "steel_plates/Faults.NNA"),
    "aps": os.path.join(EXTERNAL_DIR, "aps/aps_failure_train.csv"),
    "cmaps": os.path.join(EXTERNAL_DIR, "cmaps/train_FD001.txt"),
}
RANDOM_STATE = 42
TEST_SIZE = 0.2
SMOTE_K = 5  # For imbalance

# Model params
LOG_C = [0.01, 0.1, 1, 10]
LINEAR_ALPHA = 1.0  # For elastic net
TARGET_AUC = 0.80
