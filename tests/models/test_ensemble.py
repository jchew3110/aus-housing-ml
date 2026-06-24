"""Tests for the EnsembleHousingModel."""

import numpy as np
import pytest

from src.features.pipeline import FeaturePipeline
from src.models.ensemble import EnsembleHousingModel
from src.models.lgbm_model import LGBMHousingModel
from src.models.ridge import RidgeHousingModel
from src.models.xgboost_model import XGBoostHousingModel


@pytest.fixture
def trained_ensemble(feature_df, split_config):
    pipeline = FeaturePipeline()
    X_train, y_train = pipeline.get_X_y(feature_df, "train", split_config)
    X_val, y_val = pipeline.get_X_y(feature_df, "val", split_config)

    # Use Ridge (x2) as a cheap stand-in for constituent models
    m1 = RidgeHousingModel()
    m1.fit(X_train, y_train, X_val, y_val)
    m2 = RidgeHousingModel()
    m2.fit(X_train, y_train, X_val, y_val)

    ensemble = EnsembleHousingModel(models=[m1, m2])
    ensemble.fit(X_train, y_train, X_val, y_val)
    return ensemble, X_val, y_val


class TestEnsembleHousingModel:
    def test_weights_sum_to_one(self, trained_ensemble):
        ensemble, _, _ = trained_ensemble
        assert abs(sum(ensemble.weights) - 1.0) < 1e-9

    def test_predict_length(self, trained_ensemble):
        ensemble, X_val, y_val = trained_ensemble
        preds = ensemble.predict(X_val)
        assert len(preds) == len(y_val)

    def test_predict_no_nan(self, trained_ensemble):
        ensemble, X_val, _ = trained_ensemble
        preds = ensemble.predict(X_val)
        assert not np.any(np.isnan(preds))

    def test_predict_with_interval_ordered(self, trained_ensemble):
        ensemble, X_val, _ = trained_ensemble
        preds, lower, upper = ensemble.predict_with_interval(X_val)
        assert np.all(lower <= preds + 1e-9)
        assert np.all(upper >= preds - 1e-9)

    def test_feature_importance_length(self, trained_ensemble, feature_df, split_config):
        ensemble, _, _ = trained_ensemble
        pipeline = FeaturePipeline()
        importance = ensemble.get_feature_importance()
        assert len(importance) == len(pipeline.feature_cols)
