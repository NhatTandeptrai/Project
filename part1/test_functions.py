import pytest
import math
from unittest.mock import patch
from ols_implementation import ols_fit, hat_matrix, model_metrics, coef_inference, vif
from ridge_lasso import ridge_fit
from residual_analysis import residual_plots
from cross_validation import kfold_cv
import random
import os

@pytest.fixture
def data_1d():
    X = [[1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0], [1.0, 5.0]]
    y = [5.1, 7.9, 11.2, 13.8, 17.1]
    return X, y

@pytest.fixture
def data_2d():
    X2 = [[1.0, 1.0, 2.0], [1.0, 2.0, 1.0], [1.0, 3.0, 4.0], [1.0, 4.0, 3.0], [1.0, 5.0, 5.0]]
    y2 = [5.0, 7.0, 10.0, 12.0, 15.0]
    return X2, y2

@pytest.fixture
def data_cv():
    random.seed(42)
    X3 = [[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0], [1.0, 5.0], [1.0, 6.0], [1.0, 7.0], [1.0, 8.0], [1.0, 9.0]]
    y3 = [1.9855909670422072, 4.982709639966848, 7.9888684138432335, 11.070198372509886, 13.987241171621712, 16.850264658565905, 20.03323183440677, 22.97326625215028, 25.97830413158548, 29.011588478670085]
    return X3, y3

# 1. ols_fit
def test_ols_fit_results(data_1d):
    X, y = data_1d
    beta_hat, sigma2 = ols_fit(X, y)
    assert beta_hat[0] == pytest.approx(2.050000000000002)
    assert beta_hat[1] == pytest.approx(2.9899999999999993)
    assert sigma2 == pytest.approx(0.03566666666666664)

def test_ols_fit_error():
    with pytest.raises(ValueError, match="Không đủ bậc tự do"):
        ols_fit([[1.0, 1.0]], [5.0])

# 2. hat_matrix
def test_hat_matrix_results(data_1d):
    X, y = data_1d
    H, is_idem = hat_matrix(X)
    h_diag = [H[i][i] for i in range(len(H))]
    assert is_idem is True
    assert h_diag == pytest.approx([0.6000000000000003, 0.30000000000000016, 0.2, 0.30000000000000004, 0.6000000000000001])

def test_hat_matrix_singular():
    X_sing = [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]
    with pytest.raises(ValueError):
        hat_matrix(X_sing)

# 3. model_metrics
def test_model_metrics_results(data_1d):
    X, y = data_1d
    beta_hat, _ = ols_fit(X, y)
    y_hat = [beta_hat[0] + beta_hat[1]*X[i][1] for i in range(5)]
    metrics = model_metrics(y, y_hat, 2)
    assert metrics["R2"] == pytest.approx(0.9988045761272736)
    assert metrics["Adj_R2"] == pytest.approx(0.9984061015030314)
    assert metrics["F_statistic"] == pytest.approx(2506.5700934579468)

def test_model_metrics_constant_y():
    y_const = [5.0, 5.0, 5.0, 5.0, 5.0]
    y_hat = [5.0, 5.0, 5.0, 5.0, 5.0]
    metrics = model_metrics(y_const, y_hat, 2)
    assert math.isnan(metrics["R2"])

# 4. coef_inference
def test_coef_inference_results(data_1d):
    X, y = data_1d
    beta_hat, sigma2 = ols_fit(X, y)
    res = coef_inference(X, y, beta_hat, sigma2)
    assert res[0]["t_statistic"] == pytest.approx(10.349664149304495)
    assert res[1]["p_value"] == pytest.approx(1.7547997841038665e-05)

def test_coef_inference_error(data_1d):
    X, y = data_1d
    beta_hat, sigma2 = ols_fit(X, y)
    with pytest.raises(ValueError):
        coef_inference([[1.0, 1.0], [1.0, 2.0]], [1, 2], beta_hat, sigma2)

# 5. vif
def test_vif_results(data_2d):
    X2, y2 = data_2d
    vifs = vif(X2)
    assert vifs["Feature_Col_1"] == pytest.approx(2.7777777777777777)
    assert vifs["Feature_Col_2"] == pytest.approx(2.7777777777777777)

def test_vif_single_feature(data_1d):
    X, y = data_1d
    vifs = vif(X)
    assert len(vifs) == 1
    assert "Feature_Col_1" in vifs

# 6. ridge_fit
def test_ridge_fit_results(data_1d):
    X, y = data_1d
    beta_hat = ridge_fit(X, y, 1.0)
    assert beta_hat[0] == pytest.approx(2.865454545454547)
    assert beta_hat[1] == pytest.approx(2.718181818181815)

def test_ridge_fit_zero_lambda(data_1d):
    X, y = data_1d
    beta_ridge = ridge_fit(X, y, 0.0)
    beta_ols, _ = ols_fit(X, y)
    assert beta_ridge == pytest.approx(beta_ols)

# 7. residual_plots
@patch("builtins.print")
def test_residual_plots_results(mock_print, data_1d):
    X, y = data_1d
    beta_hat, sigma2 = ols_fit(X, y)
    savepath = "test_res.png"
    res = residual_plots(X, y, beta_hat, savepath=savepath)
    assert res["residuals"] == pytest.approx([0.05999999999999872, -0.13000000000000078, 0.17999999999999794, -0.20999999999999908, 0.10000000000000497])
    assert res["leverage"] == pytest.approx([0.6000000000000003, 0.30000000000000016, 0.2, 0.30000000000000004, 0.6000000000000001])
    if os.path.exists(savepath):
        os.remove(savepath)

def test_residual_plots_error(data_1d):
    X, y = data_1d
    beta_hat, _ = ols_fit(X, y)
    with pytest.raises(ValueError):
        residual_plots([[1, 2], [1, 3]], [1, 2], beta_hat)

# 8. kfold_cv
def test_kfold_cv_results(data_cv):
    X3, y3 = data_cv
    res = kfold_cv(X3, y3, k=2, seed=42)
    assert res["cv_score"] == pytest.approx(0.00442989549232909)
    assert len(res["fold_mses"]) == 2

def test_kfold_cv_invalid_k(data_1d):
    X, y = data_1d
    with pytest.raises(AssertionError):
        kfold_cv(X, y, k=10)
