"""
Training pipeline: split → fit all models → evaluate → save to registry.
After individual models are trained, an ensemble of XGBoost + LightGBM
is built using inverse-MAE weighting on the validation set.
"""

import logging

import pandas as pd

from src.data.config import SplitConfig
from src.features.pipeline import FeaturePipeline
from src.models.base import BaseHousingModel
from src.models.ensemble import EnsembleHousingModel
from src.models.evaluator import ModelMetrics, compare_models, evaluate
from src.models.lgbm_model import LGBMHousingModel
from src.models.registry import ModelRegistry
from src.models.ridge import RidgeHousingModel
from src.models.xgboost_model import XGBoostHousingModel

logger = logging.getLogger(__name__)

MODEL_REGISTRY: dict[str, type[BaseHousingModel]] = {
    "ridge": RidgeHousingModel,
    "xgboost": XGBoostHousingModel,
    "lgbm": LGBMHousingModel,
}


def run_training_pipeline(
    feature_df: pd.DataFrame,
    split_config: SplitConfig | None = None,
    models_to_train: list[str] | None = None,
    n_trials: int = 50,
    registry: ModelRegistry | None = None,
    train_ensemble: bool = True,
) -> dict[str, dict[str, ModelMetrics]]:
    """
    Full training pipeline.

    1. Time-based split of feature_df into train/val/test
    2. Fit each requested model (val set for tuning only — never for test eval)
    3. Evaluate on val and test
    4. Build an XGBoost+LightGBM ensemble (if both were trained and train_ensemble=True)
    5. Save all artifacts to registry
    6. Return {model_name: {split: ModelMetrics}}
    """
    if split_config is None:
        split_config = SplitConfig()
    if models_to_train is None:
        models_to_train = ["ridge", "xgboost", "lgbm"]
    if registry is None:
        registry = ModelRegistry()

    pipeline = FeaturePipeline()

    X_train, y_train = pipeline.get_X_y(feature_df, "train", split_config)
    X_val, y_val = pipeline.get_X_y(feature_df, "val", split_config)
    X_test, y_test = pipeline.get_X_y(feature_df, "test", split_config)

    logger.info(
        "Split sizes — train: %d, val: %d, test: %d",
        len(y_train),
        len(y_val),
        len(y_test),
    )

    all_results: dict[str, dict[str, ModelMetrics]] = {}
    trained_models: dict[str, BaseHousingModel] = {}

    for model_name in models_to_train:
        if model_name not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model: {model_name!r}. Choose from {list(MODEL_REGISTRY)}")

        logger.info("Training %s...", model_name)
        kwargs: dict = {}
        if model_name in ("xgboost", "lgbm"):
            kwargs["n_trials"] = n_trials

        model = MODEL_REGISTRY[model_name](**kwargs)
        model.fit(X_train, y_train, X_val, y_val)
        trained_models[model_name] = model

        splits_metrics: dict[str, ModelMetrics] = {}
        for split_name, X_split, y_split in [
            ("train", X_train, y_train),
            ("val", X_val, y_val),
            ("test", X_test, y_test),
        ]:
            if len(X_split) == 0:
                logger.info("  %s/%s — skipped (empty split)", model_name, split_name)
                continue
            preds = model.predict(X_split)
            splits_metrics[split_name] = evaluate(y_split, preds, split_name)
            m = splits_metrics[split_name]
            logger.info(
                "  %s/%s — MAE=%.3f RMSE=%.3f R²=%.3f DirAcc=%.1f%%",
                model_name,
                split_name,
                m.mae,
                m.rmse,
                m.r2,
                m.directional_accuracy * 100,
            )

        all_results[model_name] = splits_metrics
        registry.save(model, splits_metrics, pipeline.feature_cols)

    # Build ensemble from XGBoost + LightGBM if both were trained
    if (
        train_ensemble
        and "xgboost" in trained_models
        and "lgbm" in trained_models
    ):
        logger.info("Building ensemble (XGBoost + LightGBM)...")
        ensemble = EnsembleHousingModel(
            models=[trained_models["xgboost"], trained_models["lgbm"]]
        )
        ensemble.fit(X_train, y_train, X_val, y_val)

        ensemble_metrics: dict[str, ModelMetrics] = {}
        for split_name, X_split, y_split in [
            ("train", X_train, y_train),
            ("val", X_val, y_val),
            ("test", X_test, y_test),
        ]:
            if len(X_split) == 0:
                continue
            preds = ensemble.predict(X_split)
            ensemble_metrics[split_name] = evaluate(y_split, preds, split_name)
            m = ensemble_metrics[split_name]
            logger.info(
                "  ensemble/%s — MAE=%.3f RMSE=%.3f R²=%.3f DirAcc=%.1f%%",
                split_name,
                m.mae,
                m.rmse,
                m.r2,
                m.directional_accuracy * 100,
            )

        all_results["ensemble"] = ensemble_metrics
        registry.save(ensemble, ensemble_metrics, pipeline.feature_cols)

    logger.info("\nModel comparison (test split):\n%s", compare_models(all_results))
    return all_results
