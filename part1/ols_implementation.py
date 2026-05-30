import linalg_utils as tm
# 1. ols_fit
def ols_fit(X, y):
    n, num_params = len(X), len(X[0])
    if n <= num_params:
        raise ValueError(f"Không đủ bậc tự do: Số quan sát (n={n}) phải lớn hơn số tham số (p={num_params}).")
    XT = tm.transpose(X)
    XTX = tm.mat_mul(XT, X)
    XTy = tm.mat_mul(XT, [[yi] for yi in y])
    beta_hat = tm.solve_linear_system(XTX, XTy) 
    y_hat = [sum(X[i][j] * beta_hat[j][0] for j in range(num_params)) for i in range(n)]
    rss = sum((y[i] - y_hat[i])**2 for i in range(n))
    sigma2 = rss / (n - num_params)
    return [b[0] for b in beta_hat], sigma2

# 2. hat_matrix
def hat_matrix(X):
    XT = tm.transpose(X)
    XTX = tm.mat_mul(XT, X) 
    # Tính X * (X^T * X)^-1 thông qua việc giải hệ phương trình
    # Định nghĩa bài toán: XTX * M = XT -> M = (XTX)^-1 * XT
    try:
        M = tm.solve_linear_system(XTX, XT)
    except ValueError:
        raise ValueError("Ma trận X^T * X suy biến, không thể tính Hat Matrix.")
    H = tm.mat_mul(X, M)
    # Kiểm tra idempotent: H*H = H
    H2 = tm.mat_mul(H, H)
    is_idempotent = all(abs(H2[i][j] - H[i][j]) < 1e-9 for i in range(len(H)) for j in range(len(H[0])))
    return H, is_idempotent

# 3. model_metrics 
def model_metrics(y, y_hat, num_params, alpha=0.05):
    n = len(y)
    df_model = num_params - 1  # Bậc tự do của mô hình (không tính intercept)
    df_resid = n - num_params  # Bậc tự do của sai số
    
    if df_resid <= 0:
        raise ValueError(f"Không đủ bậc tự do để tính toán metrics (n={n}, p={num_params}).")
    y_mean = sum(y) / n
    rss = sum((y[i] - y_hat[i])**2 for i in range(n))
    tss = sum((y[i] - y_mean)**2 for i in range(n))
    if tss == 0:
        r2 = float('nan')
        adj_r2 = float('nan')
        f_stat = float('nan')
        f_pvalue = float('nan')
        reject_H0 = False
        comment = "Biến phụ thuộc y là hằng số (TSS = 0). Mô hình không xác định."
    elif rss == 0:
        r2 = 1.0
        adj_r2 = 1.0
        f_stat = float('inf')
        f_pvalue = 0.0
        reject_H0 = True if df_model > 0 else False
        comment = "Mô hình khớp hoàn hảo (RSS = 0)."
    else:
        r2 = 1 - (rss / tss)
        adj_r2 = 1 - (1 - r2) * (n - 1) / df_resid
        if df_model <= 0:
            f_stat = float('nan')
            f_pvalue = float('nan')
            reject_H0 = False
            comment = "Mô hình chỉ có cột Intercept, không thể thực hiện kiểm định F."
        else:
            f_stat = ( (tss - rss) / df_model ) / (rss / df_resid)
            # Tính toán p-value cho kiểm định F (Vấn đề 2)
            f_pvalue = 1.0 - tm.f_cdf(f_stat, df_model, df_resid)
            reject_H0 = f_pvalue < alpha
            comment = "Thống kê F hoàn chỉnh."
    return {
        "RSS": rss, 
        "TSS": tss, 
        "R2": r2, 
        "Adj_R2": adj_r2, 
        "F_statistic": f_stat,
        "F_pvalue": f_pvalue,
        "F_df1 (Model)": df_model,
        "F_df2 (Resid)": df_resid,
        "F_reject_H0": reject_H0,
        "Hypothesis": "H0: Tất cả hệ số hồi quy (trừ intercept) đều bằng 0",
        "Conclusion": "Bác bỏ H0, mô hình có ý nghĩa thống kê" if reject_H0 else "Chưa đủ cơ sở bác bỏ H0",
        "Note": comment
    }

# 4. coef_inference 
def coef_inference(X, y, beta_hat, sigma2, alpha=0.05):
    n, num_params = len(X), len(X[0])
    df = n - num_params
    if df <= 0:
        raise ValueError("Bậc tự do không hợp lệ (df <= 0), không thể kiểm định hệ số.")
    XTX = tm.mat_mul(tm.transpose(X), X)
    try:
        XTX_inv = tm.invert_matrix(XTX)
    except ValueError:
        raise ValueError("Ma trận X^T * X không khả nghịch, không thể tính sai số chuẩn (SE).") 
    var_beta = [[sigma2 * XTX_inv[i][j] for j in range(num_params)] for i in range(num_params)]
    # Tìm t tới hạn
    t_crit = tm.get_t_crit(alpha, df)
    inference_results = []
    for i in range(num_params):
        var_ii = var_beta[i][i]
        # Xử lý se = 0 
        se = (max(0.0, var_ii)) ** 0.5
        if se == 0:
            t_stat = float('inf') if beta_hat[i] >= 0 else float('-inf')
            p_val = 0.0
            ci_lower, ci_upper = beta_hat[i], beta_hat[i]
        else:
            t_stat = beta_hat[i] / se
            # Tính p-value 
            p_val = 2 * (1 - tm.t_cdf(abs(t_stat), df))
            ci_lower = beta_hat[i] - t_crit * se
            ci_upper = beta_hat[i] + t_crit * se
        inference_results.append({
            "Param_Index": i,
            "Coefficient": beta_hat[i],
            "Std_Err": se,
            "t_statistic": t_stat,
            "p_value": p_val,
            "Conf_Interval": (ci_lower, ci_upper),
            "Significant": p_val < alpha
        })

    return inference_results

# 5. vif 
def vif(X):
    n = len(X)
    p_orig = len(X[0])
    is_first_col_intercept = all(abs(row[0] - 1.0) < 1e-9 for row in X)
    if is_first_col_intercept:
        feature_indices = list(range(1, p_orig))
    else:
        feature_indices = list(range(0, p_orig))
    if not feature_indices:
        return [] # Không có biến độc lập nào để tính VIF ngoài intercept
    vifs = {}
    for idx in feature_indices:
        # Biến độc lập thứ idx đóng vai trò làm biến phụ thuộc tạm thời y_i
        y_i = [row[idx] for row in X]
        # X_others chứa cột Intercept (toàn 1) + tất cả các biến còn lại
        X_others = []
        for row in X:
            new_row = [1.0] 
            for j in range(len(row)):
                if j != idx and (not is_first_col_intercept or j != 0):
                    new_row.append(row[j])
            X_others.append(new_row)
        # Hồi quy y_i theo X_others
        try:
            b_i, _ = ols_fit(X_others, y_i)
            # Dự đoán
            y_i_hat = [sum(X_others[k][j] * b_i[j] for j in range(len(b_i))) for k in range(n)]
            
            y_i_mean = sum(y_i) / n
            rss_i = sum((y_i[k] - y_i_hat[k])**2 for k in range(n))
            tss_i = sum((y_i[k] - y_i_mean)**2 for k in range(n))
            if tss_i == 0:
                vif_val = float('nan') 
            elif rss_i == 0:
                vif_val = float('inf') 
            else:
                r2_i = 1.0 - (rss_i / tss_i)
                # Tránh lỗi chia cho 0 nếu R2_i vô tình bằng 1 do sai số dấu phẩy động
                vif_val = float('inf') if abs(1.0 - r2_i) < 1e-12 else 1.0 / (1.0 - r2_i)
        except ValueError:
            vif_val = float('inf')
        vifs[f"Feature_Col_{idx}"] = vif_val
    return vifs