#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
End-to-End Pipeline for Automotive Defect Prediction.
Run: python pipeline/main_pipeline.py
"""

import os
from src.dataset import load_secom, preprocess_df
from src.features import engineer_lots
from src.models import DefectPredictor
from src.plots import plot_residuals, plot_roc_curve
from src.config import PROCESSED_DIR

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs("reports/figures", exist_ok=True)

# Load and preprocess
df = load_secom()
df = preprocess_df(df, os.path.join(PROCESSED_DIR, "processed_secom.csv"))
lot_df = engineer_lots(df)

# Model
predictor = DefectPredictor()
simple_results = predictor.simple_regression(lot_df)
rmse, mae, summary = predictor.multiple_regression(lot_df)
roc_auc, pr_auc, cv_mean, cv_std = predictor.logistic_regression(df)

print("Simple Regression Top Factors:", simple_results[:5])
print(f"Multiple Regression: RMSE={rmse:.3f}, MAE={mae:.3f}")
print(
    f"Logistic: ROC-AUC={roc_auc:.3f}, PR-AUC={pr_auc:.3f}, CV={cv_mean:.3f}±{cv_std:.3f}"
)
print("AUC Target Met:", roc_auc >= 0.80)

# Plots (example for multi; adapt for roc)
# Assume y_lot_test, y_lot_pred from model
plot_residuals(y_lot_test, y_lot_pred)  # Pseudo; integrate from model
predictor.save_models()
