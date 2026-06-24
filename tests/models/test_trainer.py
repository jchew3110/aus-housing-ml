"""Tests for training pipeline and time-split correctness."""

import numpy as np
import pytest

from src.features.pipeline import FeaturePipeline
from src.models.ridge import RidgeHousingModel


class TestTimeSplit:
    def test_train_max_period_less_than_val_min(self, feature_df, split_config):
        pipeline = FeaturePipeline()
        X_train, _ = pipeline.get_X_y(feature_df, "train", split_config)
        X_val, _ = pipeline.get_X_y(feature_df, "val", split_config)

        # feature_df must have period column
        train_periods = feature_df.loc[X_train.index, "period"]
        val_periods = feature_df.loc[X_val.index, "period"]

        assert train_periods.max() < val_periods.min()

    def test_val_max_period_less_than_test_min(self, feature_df, split_config):
        pipeline = FeaturePipeline()
        X_val, _ = pipeline.get_X_y(feature_df, "val", split_config)
        X_test, _ = pipeline.get_X_y(feature_df, "test", split_config)

        val_periods = feature_df.loc[X_val.index, "period"]
        test_periods = feature_df.loc[X_test.index, "period"]

        assert val_periods.max() < test_periods.min()

    def test_splits_are_disjoint(self, feature_df, split_config):
        pipeline = FeaturePipeline()
        X_train, _ = pipeline.get_X_y(feature_df, "train", split_config)
        X_val, _ = pipeline.get_X_y(feature_df, "val", split_config)
        X_test, _ = pipeline.get_X_y(feature_df, "test", split_config)

        train_idx = set(X_train.index)
        val_idx = set(X_val.index)
        test_idx = set(X_test.index)

        assert train_idx.isdisjoint(val_idx)
        assert train_idx.isdisjoint(test_idx)
        assert val_idx.isdisjoint(test_idx)

    def test_invalid_split_raises(self, feature_df):
        pipeline = FeaturePipeline()
        with pytest.raises(ValueError, match="Unknown split"):
            pipeline.get_X_y(feature_df, "holdout")


class TestRidgeFit:
    def test_fit_and_predict(self, feature_df, split_config):
        pipeline = FeaturePipeline()
        X_train, y_train = pipeline.get_X_y(feature_df, "train", split_config)
        X_val, y_val = pipeline.get_X_y(feature_df, "val", split_config)

        model = RidgeHousingModel()
        model.fit(X_train, y_train, X_val, y_val)
        preds = model.predict(X_val)

        assert len(preds) == len(y_val)
        assert not np.any(np.isnan(preds))

    def test_predict_with_interval_bounds_ordering(self, feature_df, split_config):
        pipeline = FeaturePipeline()
        X_train, y_train = pipeline.get_X_y(feature_df, "train", split_config)
        X_val, y_val = pipeline.get_X_y(feature_df, "val", split_config)

        model = RidgeHousingModel()
        model.fit(X_train, y_train, X_val, y_val)
        preds, lower, upper = model.predict_with_interval(X_val)

        # lower ≤ point ≤ upper for all samples
        assert np.all(lower <= preds + 1e-10)
        assert np.all(upper >= preds - 1e-10)

    def test_feature_importance_length(self, trained_ridge, feature_df, split_config):
        importance = trained_ridge.get_feature_importance()
        pipeline = FeaturePipeline()
        assert len(importance) == len(pipeline.feature_cols)
