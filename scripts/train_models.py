#!/usr/bin/env python3
"""
CLI: run the training pipeline.
Usage: python scripts/train_models.py [--models ridge xgboost lgbm] [--n-trials 50]
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.data.pipeline import load_panel, run_data_pipeline
from src.features.pipeline import FeaturePipeline
from src.models.trainer import run_training_pipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def main():
    parser = argparse.ArgumentParser(description="Train housing price prediction models")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["ridge", "xgboost", "lgbm"],
        choices=["ridge", "xgboost", "lgbm", "all"],
        help="Which models to train",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Optuna trials for XGBoost/LightGBM tuning",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download data even if cached",
    )
    args = parser.parse_args()

    models = ["ridge", "xgboost", "lgbm"] if "all" in args.models else args.models

    # Load panel (run pipeline if not already done)
    try:
        panel = load_panel()
        print(f"Loaded panel from cache: {panel.shape[0]} rows")
    except FileNotFoundError:
        print("Panel not found — running data pipeline first...")
        panel = run_data_pipeline(force_download=args.force_download)

    # Build features
    print("Building feature matrix...")
    pipeline = FeaturePipeline()
    feature_df = pipeline.build(panel)
    print(f"Feature matrix: {feature_df.shape[0]} rows × {len(pipeline.feature_cols)} features")

    # Train
    print(f"\nTraining: {models} (n_trials={args.n_trials})\n")
    results = run_training_pipeline(
        feature_df,
        models_to_train=models,
        n_trials=args.n_trials,
    )

    # Summary — prefer test split, fall back to val if test is empty
    best_split = "test" if any("test" in s for s in results.values()) else "val"
    print(f"\n=== Final Results ({best_split} split) ===")
    for model_name, splits in results.items():
        split_key = "test" if "test" in splits else "val"
        m = splits[split_key]
        print(
            f"  {model_name:10s}  MAE={m.mae:.3f}  RMSE={m.rmse:.3f}  "
            f"R²={m.r2:.3f}  DirAcc={m.directional_accuracy:.1%}"
        )


if __name__ == "__main__":
    main()
