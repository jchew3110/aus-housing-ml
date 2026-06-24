#!/usr/bin/env python
"""
Generate a comprehensive model evaluation report.

Usage:
    python scripts/generate_report.py
    python scripts/generate_report.py --output reports/my_report.html

Loads all trained models from the registry, evaluates them across all splits,
and writes a self-contained HTML report with embedded charts.
"""

from __future__ import annotations

import argparse
import base64
import io
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from src.data.config import CITIES, DATA_PROCESSED_DIR, MODELS_DIR, SplitConfig
from src.features.pipeline import FEATURE_COLS, FeaturePipeline
from src.models.evaluator import calibration_coverage, evaluate, walk_forward_cv
from src.models.registry import ModelRegistry
from src.models.ridge import RidgeHousingModel

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

PALETTE = {"train": "#4c72b0", "val": "#dd8452", "test": "#55a868"}
MODEL_COLORS = ["#4c72b0", "#dd8452", "#55a868", "#c44e52"]
SPLIT_CONFIG = SplitConfig()

sns.set_theme(style="whitegrid", font_scale=1.0)
plt.rcParams.update({"figure.dpi": 130, "axes.titlesize": 11, "axes.labelsize": 10})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _b64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    enc = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return enc


def _img(b64: str, caption: str = "") -> str:
    cap = f'<p class="caption">{caption}</p>' if caption else ""
    return f'<div class="figure"><img src="data:image/png;base64,{b64}">{cap}</div>'


def _card(label: str, value: str, sub: str = "") -> str:
    sub_html = f'<div class="card-sub">{sub}</div>' if sub else ""
    return f'<div class="card"><div class="card-val">{value}</div><div class="card-lbl">{label}</div>{sub_html}</div>'


def _color_cell(val: float, low_good: bool = True, lo: float = 0.0, hi: float = 1.0) -> str:
    if val is None or np.isnan(val):
        return "<td>—</td>"
    frac = (val - lo) / (hi - lo + 1e-9)
    if low_good:
        frac = 1 - frac
    frac = max(0.0, min(1.0, frac))
    r = int(255 * (1 - frac))
    g = int(200 * frac + 55)
    b = 80
    return f'<td style="background:rgb({r},{g},{b});color:#fff;font-weight:600">{val:.4f}</td>'


def _section(title: str, body: str, interp: str = "") -> str:
    interp_html = f'<div class="interp">{interp}</div>' if interp else ""
    return (
        f'<section><h2>{title}</h2>'
        f'{interp_html}'
        f'{body}'
        f'</section>'
    )


# ---------------------------------------------------------------------------
# Data + model loading
# ---------------------------------------------------------------------------

def load_feature_df() -> pd.DataFrame:
    raw_panel = DATA_PROCESSED_DIR / "features_panel.parquet"
    log.info("Loading panel from %s", raw_panel)
    panel = pd.read_parquet(raw_panel)
    pipeline = FeaturePipeline()
    df = pipeline.build(panel)
    log.info("Feature matrix: %s rows × %s cols", *df.shape)
    return df


def load_models(registry: ModelRegistry) -> dict:
    models = {}
    for model_name in ["ridge", "xgboost", "lgbm", "ensemble"]:
        try:
            model, meta = registry.load(model_name)
            models[model_name] = {"model": model, "meta": meta}
            log.info("Loaded %s", model_name)
        except FileNotFoundError:
            log.warning("No saved model for %s — skipping", model_name)
    return models


# ---------------------------------------------------------------------------
# Prediction computation
# ---------------------------------------------------------------------------

def compute_all_predictions(
    feature_df: pd.DataFrame, models: dict
) -> dict[str, dict[str, tuple]]:
    """
    Returns {model_name: {split: (y_true, y_pred, lower, upper, periods, cities)}}
    """
    pipeline = FeaturePipeline()
    result: dict[str, dict] = {}

    for split in ("train", "val", "test"):
        mask = _split_mask(feature_df, split)
        split_df = feature_df[mask]
        periods = split_df["period"].values
        cities = split_df["city"].values if "city" in split_df.columns else np.array(["unknown"] * len(split_df))

    for name, bundle in models.items():
        model = bundle["model"]
        result[name] = {}
        for split in ("train", "val", "test"):
            mask = _split_mask(feature_df, split)
            split_df = feature_df[mask]
            X = split_df[FEATURE_COLS]
            y = split_df["target"]
            periods = split_df["period"].values
            cities = split_df["city"].values if "city" in split_df.columns else np.array(["unknown"] * len(split_df))
            try:
                preds, lo, hi = model.predict_with_interval(X, confidence=0.90)
            except Exception:
                preds = model.predict(X)
                lo, hi = preds, preds
            result[name][split] = (y.values, preds, lo, hi, periods, cities)

    return result


def _split_mask(df: pd.DataFrame, split: str) -> pd.Series:
    period = pd.PeriodIndex(df["period"], freq="Q-DEC")
    train_end = pd.Period(SPLIT_CONFIG.train_end, freq="Q-DEC")
    val_end = pd.Period(SPLIT_CONFIG.val_end, freq="Q-DEC")
    if split == "train":
        return period <= train_end
    elif split == "val":
        return (period > train_end) & (period <= val_end)
    else:
        return period > val_end


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def section_summary(feature_df: pd.DataFrame, models: dict) -> str:
    pipeline = FeaturePipeline()
    n_total = len(feature_df)
    n_train = _split_mask(feature_df, "train").sum()
    n_val = _split_mask(feature_df, "val").sum()
    n_test = _split_mask(feature_df, "test").sum()
    n_features = len(FEATURE_COLS)
    n_models = len(models)

    period_min = feature_df["period"].min()
    period_max = feature_df["period"].max()

    cards = (
        _card("Total samples", str(n_total), f"{period_min} → {period_max}")
        + _card("Training samples", str(n_train), f"≤ {SPLIT_CONFIG.train_end}")
        + _card("Validation samples", str(n_val), f"{SPLIT_CONFIG.train_end} → {SPLIT_CONFIG.val_end}")
        + _card("Test samples", str(n_test), f"> {SPLIT_CONFIG.val_end}")
        + _card("Features", str(n_features))
        + _card("Models", str(n_models))
        + _card("Cities", str(len(CITIES)))
    )
    interp = (
        "The dataset covers Australian residential property price index (RPPI) data from all 8 "
        "capital cities. The <strong>target</strong> is next-quarter QoQ % price change — a noisy, "
        "low-autocorrelation series that is genuinely hard to predict. A strictly temporal split "
        "(no shuffling) prevents future data from leaking into training, which is the single most "
        "important correctness requirement for financial time-series models."
    )
    return _section("Dataset Overview", f'<div class="cards">{cards}</div>', interp)


def section_leaderboard(preds: dict) -> str:
    rows = []
    for name, splits in preds.items():
        y_true, y_pred, lo, hi, _, _ = splits["test"]
        m = evaluate(y_true, y_pred)
        cov = calibration_coverage(y_true, lo, hi) if not np.all(lo == hi) else None
        rows.append({
            "Model": name,
            "Test MAE": m.mae,
            "Test RMSE": m.rmse,
            "Test R²": m.r2,
            "Directional Acc.": m.directional_accuracy,
            "90% Coverage": cov,
        })
    rows.sort(key=lambda r: r["Test MAE"])

    all_maes = [r["Test MAE"] for r in rows]
    all_rmses = [r["Test RMSE"] for r in rows]
    all_das = [r["Directional Acc."] for r in rows]

    header = "<tr><th>Rank</th><th>Model</th><th>Test MAE</th><th>Test RMSE</th><th>Test R²</th><th>Directional Accuracy</th><th>90% CI Coverage</th></tr>"
    body = ""
    for i, r in enumerate(rows):
        cov_val = f'{r["90% Coverage"]:.3f}' if r["90% Coverage"] is not None else "—"
        mae_cell = _color_cell(r["Test MAE"], low_good=True, lo=min(all_maes), hi=max(all_maes))
        rmse_cell = _color_cell(r["Test RMSE"], low_good=True, lo=min(all_rmses), hi=max(all_rmses))
        r2_cell = _color_cell(r["Test R²"], low_good=False, lo=-6.0, hi=1.0)
        da_cell = _color_cell(r["Directional Acc."], low_good=False, lo=min(all_das), hi=max(all_das))
        body += (
            f"<tr><td>{i+1}</td><td><strong>{r['Model']}</strong></td>"
            f"{mae_cell}{rmse_cell}{r2_cell}{da_cell}"
            f"<td>{cov_val}</td></tr>"
        )

    table = f'<table class="leaderboard"><thead>{header}</thead><tbody>{body}</tbody></table>'
    interp = (
        "<strong>MAE</strong> (mean absolute error) is the primary ranking metric — it measures "
        "the average prediction error in percentage-point terms. <strong>RMSE</strong> penalises "
        "large errors more heavily. <strong>R²</strong> measures variance explained; negative R² "
        "means the model is worse than predicting the mean (common on short, volatile series). "
        "<strong>Directional Accuracy</strong> asks: does the model at least get the direction "
        "(up/down) right? This is often more actionable than MAE for investment decisions. "
        "<strong>90% CI Coverage</strong> is the empirical fraction of actuals that fell inside "
        "the 90% prediction interval — ideally close to 0.90."
    )
    return _section("Model Leaderboard (Test Set)", table, interp)


def section_split_comparison(preds: dict) -> str:
    model_names = list(preds.keys())
    splits = ["train", "val", "test"]
    x = np.arange(len(model_names))
    width = 0.25

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric, label in zip(axes, ["mae", "rmse"], ["MAE", "RMSE"]):
        for j, split in enumerate(splits):
            vals = []
            for name in model_names:
                y_true, y_pred, *_ = preds[name][split]
                m = evaluate(y_true, y_pred)
                vals.append(getattr(m, metric))
            ax.bar(x + j * width, vals, width, label=split.capitalize(), color=PALETTE[split], alpha=0.85)
        ax.set_xticks(x + width)
        ax.set_xticklabels(model_names, rotation=15)
        ax.set_ylabel(label)
        ax.set_title(f"{label} by Split")
        ax.legend()
    fig.tight_layout()

    interp = (
        "Comparing metrics across splits reveals whether a model <em>generalises</em> or "
        "<em>memorises</em>. A model that scores well on training but poorly on validation "
        "is overfitting. Ideally, val and test performance should be similar — a large gap "
        "between them suggests the model is sensitive to the specific time period."
    )
    return _section("Performance Across Splits", _img(_b64(fig)), interp)


def section_scatter(preds: dict) -> str:
    non_ensemble = [n for n in preds if n != "ensemble"]
    ncols = 2
    nrows = (len(non_ensemble) + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 4 * nrows), squeeze=False)
    axes = axes.flatten()

    for i, name in enumerate(non_ensemble):
        y_true, y_pred, *_ = preds[name]["test"]
        ax = axes[i]
        ax.scatter(y_true, y_pred, alpha=0.6, s=25, color=MODEL_COLORS[i])
        lim = max(abs(y_true).max(), abs(y_pred).max()) * 1.1
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.8, label="Perfect")
        m = evaluate(y_true, y_pred)
        ax.set_title(f"{name}  (MAE={m.mae:.3f}, R²={m.r2:.3f})")
        ax.set_xlabel("Actual QoQ %")
        ax.set_ylabel("Predicted QoQ %")
        ax.legend(fontsize=8)

    for j in range(len(non_ensemble), len(axes)):
        axes[j].set_visible(False)
    fig.tight_layout()

    interp = (
        "Each point is one city-quarter observation on the <strong>test set</strong>. Points on "
        "the dashed diagonal indicate perfect prediction. Systematic bias appears as a cloud "
        "shifted above or below the diagonal. Variance (scatter width) reflects uncertainty in "
        "individual quarters. Note that QoQ% is inherently noisy — some scatter is unavoidable."
    )
    return _section("Predicted vs Actual (Test Set)", _img(_b64(fig)), interp)


def section_timeseries(feature_df: pd.DataFrame, preds: dict) -> str:
    test_mask = _split_mask(feature_df, "test")
    test_periods = feature_df[test_mask]["period"].values
    unique_periods = sorted(set(test_periods))

    # Average across all cities for each quarter
    y_true_by_period: dict = {}
    for name, splits in preds.items():
        y_true, y_pred, lo, hi, periods, _ = splits["test"]
        if not y_true_by_period:
            for p, yt in zip(periods, y_true):
                y_true_by_period.setdefault(p, []).append(yt)

    mean_actual = {p: np.mean(v) for p, v in y_true_by_period.items()}

    fig, ax = plt.subplots(figsize=(12, 4.5))
    period_labels = [str(p) for p in sorted(mean_actual)]
    ax.plot(period_labels, [mean_actual[p] for p in sorted(mean_actual)],
            "ko-", lw=2, ms=5, label="Actual (avg across cities)", zorder=5)

    for i, (name, splits) in enumerate(preds.items()):
        y_true, y_pred, lo, hi, periods, _ = splits["test"]
        mean_pred = {}
        for p, yp in zip(periods, y_pred):
            mean_pred.setdefault(p, []).append(yp)
        mean_pred = {p: np.mean(v) for p, v in mean_pred.items()}
        ax.plot(period_labels,
                [mean_pred.get(p, np.nan) for p in sorted(mean_actual)],
                "o--", color=MODEL_COLORS[i], lw=1.5, ms=4, label=name, alpha=0.8)

    ax.axhline(0, color="grey", lw=0.6, linestyle=":")
    ax.set_xlabel("Quarter")
    ax.set_ylabel("Avg QoQ % Change")
    ax.set_title("Test Period: Average Predicted vs Actual QoQ% (all cities)")
    ax.legend(loc="upper left", fontsize=9)
    plt.xticks(rotation=45)
    fig.tight_layout()

    interp = (
        "This chart averages predictions and actuals across all 8 cities per quarter. It shows "
        "whether models correctly identify the macro-level direction and magnitude of price "
        "movements. Divergence in late-2021 often reflects the unusual post-COVID demand shock, "
        "which is structurally different from the training period (2011–2018)."
    )
    return _section("Test Period: Predicted vs Actual Time Series", _img(_b64(fig)), interp)


def section_intervals(feature_df: pd.DataFrame, preds: dict) -> str:
    quantile_models = [n for n in preds if n in ("xgboost", "lgbm")]
    if not quantile_models:
        return ""

    best = quantile_models[0]
    y_true, y_pred, lo, hi, periods, cities = preds[best]["val"]

    # Pick one city for the plot
    city = "Sydney"
    mask = cities == city
    if mask.sum() == 0:
        mask = np.ones(len(cities), dtype=bool)
        city = "All"

    yt, yp, yl, yh, ps = y_true[mask], y_pred[mask], lo[mask], hi[mask], periods[mask]
    order = np.argsort([str(p) for p in ps])
    yt, yp, yl, yh, ps = yt[order], yp[order], yl[order], yh[order], ps[order]

    inside = (yt >= yl) & (yt <= yh)
    cov = float(np.mean(inside))
    labels = [str(p) for p in ps]

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.fill_between(labels, yl, yh, alpha=0.25, color="#4c72b0", label="90% PI")
    ax.plot(labels, yp, "-", color="#4c72b0", lw=2, label="Predicted")
    ax.scatter(labels, yt, c=np.where(inside, "#55a868", "#c44e52"), s=40, zorder=5,
               label="Actual (green=inside, red=outside)")
    ax.axhline(0, color="grey", lw=0.6, linestyle=":")
    ax.set_title(f"{best} — Validation set, {city}  |  Empirical 90% coverage = {cov:.1%}")
    ax.set_xlabel("Quarter")
    ax.set_ylabel("QoQ %")
    plt.xticks(rotation=45)
    ax.legend(fontsize=9)
    fig.tight_layout()

    interp = (
        f"The shaded band is the 90% prediction interval from the {best} quantile model. "
        "Green dots fell <em>inside</em> the interval; red dots fell <em>outside</em>. "
        f"Empirical coverage is {cov:.1%} — a well-calibrated interval should be close to 90%. "
        "Under-coverage (< 90%) means the intervals are too narrow and overconfident. "
        "Over-coverage (> 90%) means they are too wide and less informative."
    )
    return _section("Prediction Intervals & Calibration", _img(_b64(fig)), interp)


def section_feature_importance(preds: dict, models: dict) -> str:
    non_ensemble = [n for n in models if n != "ensemble"]
    ncols = 2
    nrows = (len(non_ensemble) + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 5 * nrows), squeeze=False)
    axes = axes.flatten()

    for i, name in enumerate(non_ensemble):
        model = models[name]["model"]
        try:
            imp = model.get_feature_importance().head(15)
        except Exception as e:
            axes[i].set_title(f"{name}: error — {e}")
            continue
        ax = axes[i]
        colors = [MODEL_COLORS[i]] * len(imp)
        ax.barh(imp.index[::-1], imp.values[::-1], color=colors, alpha=0.85)
        ax.set_title(f"{name} — Top {len(imp)} features")
        ax.set_xlabel("Importance score")

    for j in range(len(non_ensemble), len(axes)):
        axes[j].set_visible(False)
    fig.tight_layout()

    interp = (
        "Feature importance scores show which inputs influence model predictions the most. "
        "For <strong>Ridge</strong>, importance = absolute coefficient magnitude (after scaling). "
        "For <strong>tree models</strong>, it is the total gain from splitting on each feature. "
        "High importance does not mean a feature causes price changes — it means the model "
        "found it predictive. Lagged price levels dominate because of mean-reversion dynamics."
    )
    return _section("Feature Importance (Top 15)", _img(_b64(fig)), interp)


def section_shap(feature_df: pd.DataFrame, preds: dict, models: dict) -> str:
    non_ensemble = [n for n in models if n != "ensemble"]
    test_mask = _split_mask(feature_df, "test")
    X_test = feature_df[test_mask][FEATURE_COLS].head(100)  # cap at 100 rows for speed

    ncols = 2
    nrows = (len(non_ensemble) + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 5 * nrows), squeeze=False)
    axes = axes.flatten()
    shap_succeeded = False

    for i, name in enumerate(non_ensemble):
        model = models[name]["model"]
        ax = axes[i]
        try:
            shap_df = model.compute_shap(X_test)
            mean_abs = shap_df.abs().mean().sort_values(ascending=False).head(15)
            ax.barh(mean_abs.index[::-1], mean_abs.values[::-1], color=MODEL_COLORS[i], alpha=0.85)
            ax.set_title(f"{name} — Mean |SHAP| (top 15)")
            ax.set_xlabel("Mean absolute SHAP value")
            shap_succeeded = True
        except Exception as e:
            ax.text(0.5, 0.5, f"SHAP unavailable:\n{e}",
                    ha="center", va="center", transform=ax.transAxes, fontsize=9)
            ax.set_title(f"{name} — SHAP")

    for j in range(len(non_ensemble), len(axes)):
        axes[j].set_visible(False)
    fig.tight_layout()

    if not shap_succeeded:
        plt.close(fig)
        return ""

    interp = (
        "SHAP (SHapley Additive exPlanations) quantifies each feature's average contribution "
        "to pushing predictions away from the model's baseline. Unlike raw importance scores, "
        "SHAP values are <em>additive</em>: summing a row's SHAP values plus the base value "
        "reproduces the model's prediction. This makes SHAP the most trustworthy way to "
        "understand what drove a specific prediction. The bars show mean <em>absolute</em> SHAP "
        "across the test set — features at the top matter most on average."
    )
    return _section("SHAP Feature Contributions", _img(_b64(fig)), interp)


def section_city_heatmap(feature_df: pd.DataFrame, preds: dict) -> str:
    model_names = list(preds.keys())
    city_maes: dict[str, dict] = {name: {} for name in model_names}

    test_mask = _split_mask(feature_df, "test")
    test_cities = feature_df[test_mask]["city"].values if "city" in feature_df.columns else None

    if test_cities is None:
        return ""

    for name, splits in preds.items():
        y_true, y_pred, lo, hi, periods, cities = splits["test"]
        for city in CITIES:
            mask = cities == city
            if mask.sum() == 0:
                city_maes[name][city] = np.nan
            else:
                m = evaluate(y_true[mask], y_pred[mask])
                city_maes[name][city] = m.mae

    heat_df = pd.DataFrame(city_maes).T  # models × cities
    heat_df = heat_df[CITIES]

    fig, ax = plt.subplots(figsize=(12, 3.5))
    sns.heatmap(
        heat_df,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        linewidths=0.3,
        ax=ax,
        cbar_kws={"label": "MAE (QoQ pp)"},
    )
    ax.set_title("Test MAE by Model × City")
    ax.set_xlabel("City")
    ax.set_ylabel("Model")
    fig.tight_layout()

    interp = (
        "Some cities are structurally harder to predict than others. Darwin and Hobart tend "
        "to be more volatile (smaller markets, less liquidity) and show higher MAE. Sydney and "
        "Melbourne are the largest markets with the most data, but they also experience the "
        "largest boom/bust cycles. High MAE in all models for a city signals irreducible "
        "uncertainty — no model has enough signal to forecast that city reliably."
    )
    return _section("City-Level Test MAE Heatmap", _img(_b64(fig)), interp)


def section_residuals(preds: dict) -> str:
    model_names = list(preds.keys())
    ncols = 2
    nrows = (len(model_names) + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 4 * nrows), squeeze=False)
    axes = axes.flatten()

    for i, name in enumerate(model_names):
        y_true, y_pred, *_ = preds[name]["test"]
        residuals = y_true - y_pred
        ax = axes[i]
        ax.hist(residuals, bins=25, color=MODEL_COLORS[i % len(MODEL_COLORS)], alpha=0.75, density=True)
        x = np.linspace(residuals.min(), residuals.max(), 200)
        mu, sigma = np.mean(residuals), np.std(residuals)
        ax.plot(x, (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2),
                "k-", lw=1.5, label="Normal fit")
        ax.axvline(0, color="red", lw=1, linestyle="--")
        ax.set_title(f"{name}  (μ={mu:.3f}, σ={sigma:.3f})")
        ax.set_xlabel("Residual (actual − predicted, pp)")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)

    for j in range(len(model_names), len(axes)):
        axes[j].set_visible(False)
    fig.tight_layout()

    interp = (
        "Residual histograms should ideally be symmetric around zero (unbiased) and roughly "
        "Gaussian (well-calibrated error). A mean residual near zero means the model has no "
        "systematic bias. Wide tails indicate the model struggles with extreme market events. "
        "A shift of the histogram left or right reveals systematic over- or under-prediction."
    )
    return _section("Residual Distribution (Test Set)", _img(_b64(fig)), interp)


def section_cv(feature_df: pd.DataFrame) -> str:
    log.info("Running walk-forward CV for Ridge (fast baseline)...")
    try:
        fold_metrics = walk_forward_cv(
            feature_df,
            RidgeHousingModel,
            n_splits=5,
            min_train_quarters=20,
            val_quarters=8,
        )
    except Exception as e:
        log.warning("CV failed: %s", e)
        return ""

    if not fold_metrics:
        return ""

    fold_maes = [m.mae for m in fold_metrics]
    fold_r2s = [m.r2 for m in fold_metrics]
    fold_das = [m.directional_accuracy for m in fold_metrics]
    folds = [f"Fold {i+1}" for i in range(len(fold_metrics))]

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    for ax, vals, label, color in zip(
        axes,
        [fold_maes, fold_r2s, fold_das],
        ["MAE", "R²", "Directional Acc."],
        ["#4c72b0", "#dd8452", "#55a868"],
    ):
        ax.plot(folds, vals, "o-", color=color, lw=2, ms=7)
        ax.axhline(np.mean(vals), color=color, lw=1, linestyle="--", alpha=0.6,
                   label=f"Mean={np.mean(vals):.3f}")
        ax.set_title(f"CV {label} (Ridge)")
        ax.set_ylabel(label)
        ax.legend(fontsize=9)
        ax.set_ylim(bottom=0 if label in ("MAE",) else None)
    fig.tight_layout()

    mean_mae = np.mean(fold_maes)
    std_mae = np.std(fold_maes)
    interp = (
        f"Walk-forward cross-validation trains on an expanding window and evaluates on a fixed "
        f"8-quarter window. Each fold's training set is strictly older than its validation set — "
        f"no data leakage. Ridge mean CV MAE = <strong>{mean_mae:.3f} ± {std_mae:.3f}</strong>. "
        "Stable MAE across folds means the model's error is consistent over time. Rising MAE in "
        "later folds can indicate concept drift — the market regime has shifted in a way the "
        "model cannot adapt to."
    )
    return _section("Walk-Forward Cross-Validation (Ridge)", _img(_b64(fig)), interp)


# ---------------------------------------------------------------------------
# HTML assembly
# ---------------------------------------------------------------------------

CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
       background: #f5f6fa; color: #1a1a2e; line-height: 1.6; }
header { background: linear-gradient(135deg, #1a1a2e, #16213e);
         color: #fff; padding: 2.5rem 3rem; }
header h1 { font-size: 1.9rem; margin-bottom: 0.3rem; }
header p  { opacity: 0.7; font-size: 0.95rem; }
main { max-width: 1200px; margin: 2rem auto; padding: 0 1.5rem; }
section { background: #fff; border-radius: 10px; padding: 2rem;
          margin-bottom: 2rem; box-shadow: 0 2px 8px rgba(0,0,0,.07); }
section h2 { font-size: 1.25rem; margin-bottom: 1.1rem;
             padding-bottom: 0.6rem; border-bottom: 2px solid #e8ecf0; }
.interp { background: #f0f4ff; border-left: 4px solid #4c72b0;
          padding: 0.8rem 1.1rem; border-radius: 0 6px 6px 0;
          margin-bottom: 1.2rem; font-size: 0.92rem; color: #333; }
.cards  { display: flex; flex-wrap: wrap; gap: 1rem; }
.card   { background: #f0f4ff; border-radius: 8px; padding: 1rem 1.4rem;
          min-width: 130px; text-align: center; }
.card-val { font-size: 1.6rem; font-weight: 700; color: #1a1a2e; }
.card-lbl { font-size: 0.8rem; text-transform: uppercase;
            letter-spacing: 0.05em; color: #555; margin-top: 0.2rem; }
.card-sub { font-size: 0.75rem; color: #777; margin-top: 0.2rem; }
table.leaderboard { border-collapse: collapse; width: 100%; font-size: 0.9rem; }
table.leaderboard th, table.leaderboard td
      { padding: 0.55rem 0.9rem; border: 1px solid #e0e4ea; }
table.leaderboard thead { background: #1a1a2e; color: #fff; }
table.leaderboard tbody tr:nth-child(even) { background: #f8f9fc; }
.figure { text-align: center; margin-top: 1rem; }
.figure img { max-width: 100%; border-radius: 6px;
              box-shadow: 0 1px 4px rgba(0,0,0,.12); }
.caption { font-size: 0.82rem; color: #666; margin-top: 0.4rem; }
footer { text-align: center; padding: 2rem; font-size: 0.8rem; color: #888; }
"""


def build_html(sections: list[str], generated_at: str) -> str:
    body = "\n".join(s for s in sections if s)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AUS Housing ML — Evaluation Report</title>
<style>{CSS}</style>
</head>
<body>
<header>
  <h1>AUS Housing ML — Evaluation Report</h1>
  <p>Generated {generated_at} &nbsp;|&nbsp; Predicting next-quarter RPPI QoQ% across 8 Australian capital cities</p>
</header>
<main>
{body}
</main>
<footer>aus-housing-ml v0.3.0 &nbsp;·&nbsp; Temporal train/val/test split &nbsp;·&nbsp;
Walk-forward CV &nbsp;·&nbsp; SHAP explainability</footer>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate model evaluation report")
    parser.add_argument("--output", default="reports/evaluation_report.html",
                        help="Output HTML file path")
    parser.add_argument("--model-dir", default=str(MODELS_DIR))
    args = parser.parse_args()

    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    registry = ModelRegistry(models_dir=Path(args.model_dir))
    feature_df = load_feature_df()
    models = load_models(registry)

    if not models:
        log.warning("No trained models found. Train models first with: make train")
        sys.exit(1)

    log.info("Computing predictions for all models and splits...")
    preds = compute_all_predictions(feature_df, models)

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    log.info("Building report sections...")

    sections = [
        section_summary(feature_df, models),
        section_leaderboard(preds),
        section_split_comparison(preds),
        section_scatter(preds),
        section_timeseries(feature_df, preds),
        section_intervals(feature_df, preds),
        section_feature_importance(preds, models),
        section_shap(feature_df, preds, models),
        section_city_heatmap(feature_df, preds),
        section_residuals(preds),
        section_cv(feature_df),
    ]

    html = build_html(sections, generated_at)
    output_path.write_text(html, encoding="utf-8")
    log.info("Report written → %s  (%d KB)", output_path, len(html) // 1024)
    print(f"\n  Report: file://{output_path.resolve()}\n")


if __name__ == "__main__":
    main()
