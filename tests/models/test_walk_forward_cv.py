"""Tests for walk-forward cross-validation."""

import pytest

from src.models.evaluator import ModelMetrics, walk_forward_cv
from src.models.ridge import RidgeHousingModel


class TestWalkForwardCV:
    def test_returns_list_of_metrics(self, feature_df):
        folds = walk_forward_cv(
            feature_df,
            model_cls=RidgeHousingModel,
            n_splits=3,
            min_train_quarters=16,
            val_quarters=4,
        )
        assert isinstance(folds, list)
        assert len(folds) > 0
        assert all(isinstance(m, ModelMetrics) for m in folds)

    def test_each_fold_has_positive_samples(self, feature_df):
        folds = walk_forward_cv(
            feature_df,
            model_cls=RidgeHousingModel,
            n_splits=3,
            min_train_quarters=16,
            val_quarters=4,
        )
        for fold in folds:
            assert fold.n_samples > 0

    def test_directional_accuracy_in_range(self, feature_df):
        folds = walk_forward_cv(
            feature_df,
            model_cls=RidgeHousingModel,
            n_splits=3,
            min_train_quarters=16,
            val_quarters=4,
        )
        for fold in folds:
            assert 0.0 <= fold.directional_accuracy <= 1.0

    def test_raises_on_insufficient_data(self, feature_df):
        with pytest.raises(ValueError, match="Not enough data"):
            walk_forward_cv(
                feature_df,
                model_cls=RidgeHousingModel,
                n_splits=2,
                min_train_quarters=999,
                val_quarters=4,
            )
