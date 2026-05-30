import sys
import os

thu_muc_cha = os.path.abspath("..")
if thu_muc_cha not in sys.path:
    sys.path.append(thu_muc_cha)
from part1.linalg_utils import transpose, matmul, matvec, inv, mat_add, mat_scale, eye
from model_comparison import to_list_data, add_intercept, predict, regression_metrics
def bayesian_fit(X, y, sigma2 = 1.0, prior_var = 10.0):
    n = len(X)
    p = len(X[0])
    m0 = [0.0] * p
    S0_inv = mat_scale(eye(p), 1/prior_var)
    XT = transpose(X)
    XTX = matmul(XT, X)
    XTy = matvec(XT, y)
    Sn_inv = mat_add(S0_inv, mat_scale(XTX, 1/sigma2))
    Sn = inv(Sn_inv)
    prior_part = matvec(S0_inv, m0)
    data_part = [value * (1 / sigma2) for value in XTy]
    right_side = [
        prior_part[i] + data_part[i]
        for i in range(p)
    ]
    mn = matvec(Sn, right_side)
    return mn, Sn
def run_bayesian_linear_regression(X_train, y_train, X_test, y_test, sigma2=1.0, prior_var=10.0):
    X_train ,y_train = to_list_data(X_train, y_train)
    X_test, y_test = to_list_data(X_test, y_test)
    X_train = add_intercept(X_train)
    X_test = add_intercept(X_test)
    mn, Sn = bayesian_fit(X_train, y_train, sigma2, prior_var)
    y_pred = predict(X_test, mn)
    metrics = regression_metrics(y_test, y_pred)
    return {
        "model_name": "Bayesian Linear Regression",
        "posterior_mean": mn,
        "posterior_cov": Sn,
        "sigma2": sigma2,
        "prior_var": prior_var,
        "y_pred": y_pred,
        "metrics": metrics
    }
