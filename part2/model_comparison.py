import sys
import os
import math
import pandas as pd
# Lùi lại 1 cấp thư mục (ra khỏi 'part 2' để nhìn thấy 'part 1')
thu_muc_cha = os.path.abspath('..')
if thu_muc_cha not in sys.path:
    sys.path.append(thu_muc_cha)
from part1.linalg_utils import matvec
from part1.ols_implementation import ols_fit, vif
from part1.ridge_lasso import ridge_fit
from part1.cross_validation import _shuffle_indices, _split_into_folds
def to_list_data(X, y): #chuyển dữ liệu X và y sang list
    if hasattr(X, "values"):
        X_list = X.values.tolist()
    else:
        X_list = X

    if hasattr(y, "tolist"):
        y_list = y.tolist()
    else:
        y_list = y

    return X_list, y_list
def add_intercept(X):
    return [[1.0] + list(row) for row in X]
def predict(X, beta):
    return matvec(X, beta)
def regression_metrics(y_true, y_pred):
    n = len(y_true)
    y_mean = sum(y_true)/n
    RSS = sum((y_true[i] - y_pred[i])**2 for i in range(n))
    TSS = sum((y_true[i] - y_mean)**2 for i in range(n))
    MAE = sum(abs(y_true[i] - y_pred[i]) for i in range(n))/n
    RMSE = math.sqrt(RSS/n)
    R2 = 1 - RSS/TSS if TSS != 0 else 0
    return {
        'MAE': MAE,
        'RMSE': RMSE,
        'R2': R2
    }
def run_ols_basic(X_train, y_train, X_test, y_test):
    # chuyển dữ liệu
    X_test, y_test = to_list_data(X_test, y_test)
    X_train, y_train= to_list_data(X_train, y_train)
    # thêm intercept
    X_test = add_intercept(X_test)
    X_train = add_intercept(X_train)

    beta, sigma2 = ols_fit(X_train, y_train)
    y_pred = predict(X_test, beta)

    metrics = regression_metrics(y_test, y_pred)
    return {
    "model_name": "OLS Basic",
    "beta": beta,
    "sigma2": sigma2,
    "y_pred": y_pred,
    "metrics": metrics
    }
# model 2
def calculate_vif_table(X, feature_names):
    vif_value = vif(X)
    vif_table = []

    if isinstance(vif_value, dict):
        for i, feature in enumerate(feature_names):
            key = f"Feature_Col_{i + 1}"  # +1 vì X đã có intercept ở cột 0
            vif_table.append({
                "Feature": feature,
                "VIF": vif_value.get(key, float("nan"))
            })

    # Trường hợp vif() bản cũ trả về list
    else:
        for i, feature in enumerate(feature_names):
            vif_table.append({
                "Feature": feature,
                "VIF": vif_value[i]
            })

    return pd.DataFrame(vif_table).sort_values("VIF", ascending=False)

def select_features_by_vif(X_train, threshold=10):
    feature_names = X_train.columns.tolist()
    X_train, _ = to_list_data(X_train, [0] * len(X_train))
    X_train = add_intercept(X_train)
    vif_table = calculate_vif_table(X_train, feature_names)
    select_features = vif_table[vif_table['VIF'] <= threshold]['Feature'].tolist()
    if len(select_features) == 0:
        select_features = vif_table.sort_values('VIF').head(10)['Feature'].tolist()
    return select_features, vif_table
def run_ols_selected_by_vif(X_train, y_train, X_test, y_test, threshold=10):
    select_features, vif_table = select_features_by_vif(X_train, threshold)
    X_train_selected = X_train[select_features]
    X_test_selected = X_test[select_features]
    result = run_ols_basic(X_train_selected, y_train, X_test_selected, y_test)
    result['model_name'] = 'OLS Selected by VIF'
    result['selected_features'] = select_features
    result['vif_table'] = vif_table
    return result
# model 3
def kfold_cv_ridge(X, y, lambdas, k=5, seed=42): # tìm lambda tốt nhất
   n = len(X)
   indices = _shuffle_indices(n, seed)
   folds = _split_into_folds(indices, k)
   cv_results =[]
   for lam in lambdas:
       fold_mses = []
       for i in range(k):
           val_idx = folds[i]
           train_idx = []
           
           for j in range(k):
               if j != i:
                   train_idx.extend(folds[j])
           X_train_fold = [X[idx] for idx in train_idx]
           y_train_fold = [y[idx] for idx in train_idx]

           X_val_fold = [X[idx] for idx in val_idx]
           y_val_fold = [y[idx] for idx in val_idx]
           
           beta = ridge_fit(X_train_fold, y_train_fold, lam)
           y_val_pred = predict(X_val_fold, beta)
           mse = sum((y_val_fold[t] - y_val_pred[t]) ** 2 for t in range(len(y_val_fold))) / len(y_val_fold)
           fold_mses.append(mse)
       mean_mse = sum(fold_mses)/k
       cv_results.append({
            "lambda": lam,
            "CV_MSE": mean_mse
        })

   cv_results = pd.DataFrame(cv_results)
   best_lambda = cv_results.sort_values("CV_MSE").iloc[0]["lambda"]

   return best_lambda, cv_results
def run_ridge_cv(X_train, y_train, X_test, y_test, lambdas, k=5, seed=42):
    X_train, y_train = to_list_data(X_train, y_train)
    X_test, y_test = to_list_data(X_test, y_test)

    X_train = add_intercept(X_train)
    X_test = add_intercept(X_test)

    best_lambda, cv_results = kfold_cv_ridge(
        X_train,
        y_train,
        lambdas,
        k=k,
        seed=seed
    )

    beta = ridge_fit(X_train, y_train, best_lambda)
    y_pred = predict(X_test, beta)

    metrics = regression_metrics(y_test, y_pred)

    return {
        "model_name": "Ridge Regression",
        "beta": beta,
        "best_lambda": best_lambda,
        "cv_results": cv_results,
        "y_pred": y_pred,
        "metrics": metrics
    }