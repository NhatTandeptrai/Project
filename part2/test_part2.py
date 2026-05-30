"""
Unit tests cho Phần 2 của đồ án.
Chạy:
    pytest part2/test_part2.py -q

"""

import os
import sys
import math

import pandas as pd
import pytest

# Cho phép chạy test từ cả thư mục gốc Group_<ID>/ lẫn trực tiếp trong part2/
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
PART1_DIR = os.path.join(BASE_DIR, "part1")

for path in [CURRENT_DIR, BASE_DIR, PART1_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

from data_pipeline import DataPipeline
from model_comparison import (
    add_intercept,
    predict,
    regression_metrics,
    run_ols_basic,
    run_ols_selected_by_vif,
    run_ridge_cv,
)


@pytest.fixture
def raw_train_test_data():
    """Dữ liệu nhỏ có missing values, categorical, constant và duplicate columns."""
    X_train = pd.DataFrame({
        "num1": [1.0, 2.0, None, 4.0, 5.0, 6.0],
        "num2": [10.0, 20.0, 30.0, 40.0, None, 60.0],
        "dup_num2": [10.0, 20.0, 30.0, 40.0, None, 60.0],
        "cat": ["A", "B", None, "A", "C", "B"],
        "constant": [1, 1, 1, 1, 1, 1],
        "drop_me": [100, 101, 102, 103, 104, 105],
    })

    X_test = pd.DataFrame({
        "num1": [2.0, None, 7.0],
        "num2": [15.0, 25.0, None],
        "dup_num2": [15.0, 25.0, None],
        "cat": ["A", "D", None],
        "constant": [1, 1, 1],
        "drop_me": [200, 201, 202],
    })

    return X_train, X_test


@pytest.fixture
def regression_data():
    """Dữ liệu hồi quy tuyến tính nhỏ, đủ dòng để OLS/Ridge chạy ổn định."""
    X_train = pd.DataFrame({
        "x1": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        "x2": [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
    })
    y_train = pd.Series([
        1.0 + 2.0 * x1 - 0.5 * x2
        for x1, x2 in zip(X_train["x1"], X_train["x2"])
    ])

    X_test = pd.DataFrame({
        "x1": [8.0, 9.0],
        "x2": [1.0, 0.0],
    })
    y_test = pd.Series([
        1.0 + 2.0 * x1 - 0.5 * x2
        for x1, x2 in zip(X_test["x1"], X_test["x2"])
    ])

    return X_train, y_train, X_test, y_test


# Test DataPipeline
def test_pipeline_fit_transform_has_no_missing_values(raw_train_test_data):
    X_train, _ = raw_train_test_data
    pipeline = DataPipeline(drop_columns=["drop_me"])

    X_processed = pipeline.fit_transform(X_train)

    assert pipeline.is_fitted is True
    assert X_processed.isna().sum().sum() == 0
    assert "drop_me" not in X_processed.columns
    assert "constant" not in X_processed.columns
    assert "dup_num2" not in X_processed.columns


def test_pipeline_transform_keeps_same_columns_as_train(raw_train_test_data):
    X_train, X_test = raw_train_test_data
    pipeline = DataPipeline(drop_columns=["drop_me"])

    X_train_processed = pipeline.fit_transform(X_train)
    X_test_processed = pipeline.transform(X_test)

    assert list(X_test_processed.columns) == list(X_train_processed.columns)
    assert X_test_processed.shape[1] == X_train_processed.shape[1]
    assert X_test_processed.isna().sum().sum() == 0


def test_pipeline_transform_before_fit_raises_error(raw_train_test_data):
    _, X_test = raw_train_test_data
    pipeline = DataPipeline()

    with pytest.raises(ValueError, match="Pipeline chưa được fit"):
        pipeline.transform(X_test)


def test_pipeline_rejects_non_dataframe_input():
    pipeline = DataPipeline()

    with pytest.raises(TypeError, match="pandas DataFrame"):
        pipeline.fit([[1, 2], [3, 4]])


# Test model_comparison helper functions
def test_add_intercept_and_predict():
    X = [[2.0, 3.0], [4.0, 5.0]]
    X_with_intercept = add_intercept(X)

    assert X_with_intercept == [[1.0, 2.0, 3.0], [1.0, 4.0, 5.0]]
    assert predict(X_with_intercept, [1.0, 2.0, -1.0]) == pytest.approx([2.0, 4.0])


def test_regression_metrics_known_values():
    y_true = [1.0, 2.0, 3.0]
    y_pred = [1.0, 2.0, 4.0]

    metrics = regression_metrics(y_true, y_pred)

    assert metrics["MAE"] == pytest.approx(1.0 / 3.0)
    assert metrics["RMSE"] == pytest.approx(math.sqrt(1.0 / 3.0))
    assert metrics["R2"] == pytest.approx(0.5)


# Test 3 mô hình Phần 2
def test_run_ols_basic_on_simple_linear_data(regression_data):
    X_train, y_train, X_test, y_test = regression_data

    result = run_ols_basic(X_train, y_train, X_test, y_test)

    assert result["model_name"] == "OLS Basic"
    assert result["metrics"]["RMSE"] == pytest.approx(0.0, abs=1e-8)
    assert result["metrics"]["R2"] == pytest.approx(1.0)
    assert result["y_pred"] == pytest.approx(y_test.tolist())


def test_run_ols_selected_by_vif_returns_selected_features(regression_data):
    X_train, y_train, X_test, y_test = regression_data

    result = run_ols_selected_by_vif(
        X_train,
        y_train,
        X_test,
        y_test,
        threshold=10,
    )

    assert result["model_name"] == "OLS Selected by VIF"
    assert len(result["selected_features"]) >= 1
    assert set(result["selected_features"]).issubset(set(X_train.columns))
    assert {"Feature", "VIF"}.issubset(set(result["vif_table"].columns))
    assert "RMSE" in result["metrics"]


def test_run_ridge_cv_selects_lambda_and_returns_metrics(regression_data):
    X_train, y_train, X_test, y_test = regression_data
    lambdas = [0.0, 0.1, 1.0]

    result = run_ridge_cv(
        X_train,
        y_train,
        X_test,
        y_test,
        lambdas=lambdas,
        k=4,
        seed=42,
    )

    assert result["model_name"] == "Ridge Regression"
    assert result["best_lambda"] in lambdas
    assert len(result["cv_results"]) == len(lambdas)
    assert {"lambda", "CV_MSE"}.issubset(set(result["cv_results"].columns))
    assert "RMSE" in result["metrics"]
    assert len(result["y_pred"]) == len(y_test)
