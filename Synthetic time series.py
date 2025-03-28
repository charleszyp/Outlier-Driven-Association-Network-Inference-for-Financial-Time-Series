import numpy as np
import pandas as pd
from tqdm import tqdm  # 显示进度条
import time

# 设置 rho 值
rho = 0.1  # 您已将 rho 设置为 0.1

# 循环运行10次
for iteration in range(1, 11):
    # 生成一个基于当前时间的随机种子
    seed = int((time.time() * 1000) % 100000)
    np.random.seed(seed)
    print(f"Iteration {iteration}, Random Seed: {seed}")

    n = 10  # 设置时间序列数量为10

    # 初始化各参数矩阵为零矩阵
    miu_matrix = np.zeros((n, n))  # 初始化均值影响系数矩阵
    alpha_matrix = np.zeros((n, n))  # 初始化ARCH系数矩阵
    beta_matrix = np.zeros((n, n))  # 初始化GARCH系数矩阵
    alpha_0 = np.random.rand(n) * 0.05  # 0-0.05

    # 设置 miu_matrix 的非对角线元素和 alpha_matrix、beta_matrix 的所有元素
    for i in tqdm(range(n), desc="Setting matrices"):
        for j in range(n):
            if i != j:
                random_value = (np.random.rand() - 0.5) * 0.1 if np.random.rand() > 0.3 else 0
                miu_matrix[i, j] = random_value
            else:
                miu_matrix[i, j] = 0.1  # 将对角线元素设置为 0.1

    # 计算系数矩阵的特征值
    A = rho * miu_matrix
    eigenvalues = np.linalg.eigvals(A)
    max_eigenvalue = np.max(np.abs(eigenvalues))
    print(f"Maximum eigenvalue modulus before scaling: {max_eigenvalue}")

    # 如果最大特征值的模大于等于 1，则缩放系数矩阵
    if max_eigenvalue >= 1:
        scaling_factor = 0.99 / max_eigenvalue
        miu_matrix *= scaling_factor
        print(f"Scaling miu_matrix by factor {scaling_factor} to ensure stationarity.")
        # 重新计算特征值以验证
        A = rho * miu_matrix
        eigenvalues = np.linalg.eigvals(A)
        max_eigenvalue = np.max(np.abs(eigenvalues))
        print(f"Maximum eigenvalue modulus after scaling: {max_eigenvalue}")

    # 设置 alpha_matrix 和 beta_matrix，并确保参数之和小于 1
    for i in tqdm(range(n), desc="Setting alpha and beta matrices"):
        alpha_row = np.random.rand(n) * 0.05
        beta_row = np.random.rand(n) * 0.05
        sum_params = np.sum(alpha_row) + np.sum(beta_row)
        if sum_params >= 1:
            scaling_factor = 0.99 / sum_params
            alpha_row *= scaling_factor
            beta_row *= scaling_factor
        alpha_matrix[i, :] = alpha_row
        beta_matrix[i, :] = beta_row

    T = 10000  # 设置时间序列长度为 10000

    # 初始化时间序列和误差项矩阵
    y = np.zeros((T, n))
    y[0, :] = np.random.uniform(-1, 1, n)
    sigma_squared = np.full((T, n), 0.01)
    epsilon = np.zeros((T, n))
    gamma = np.zeros((T, n))

    # 设置条件方差和 gamma 的上限
    max_sigma_squared = 1e6
    max_gamma = 1e3

    # 生成时间序列
    for t in tqdm(range(1, T), desc="Generating Time Series"):
        for i in range(n):
            gamma_squared = gamma[t - 1, :] ** 2

            # 检查 gamma_squared 是否出现溢出
            gamma_squared = np.nan_to_num(gamma_squared, nan=0.0, posinf=max_gamma ** 2, neginf=-max_gamma ** 2)

            # 计算条件方差 sigma_squared[t, i]
            sigma_squared[t, i] = (
                alpha_0[i]
                + rho * (
                    np.dot(alpha_matrix[i, :], gamma_squared)
                    + np.dot(beta_matrix[i, :], sigma_squared[t - 1, :])
                )
            )

            # 防止 sigma_squared[t, i] 为负数或超过上限
            sigma_squared[t, i] = np.clip(sigma_squared[t, i], 1e-8, max_sigma_squared)

            # 生成标准正态随机数
            epsilon[t, i] = np.random.normal(0, 1)

            # 计算 gamma_{i,t}
            gamma[t, i] = np.sqrt(sigma_squared[t, i]) * epsilon[t, i]

            # 设置 gamma 的上限
            gamma[t, i] = np.clip(gamma[t, i], -max_gamma, max_gamma)

            # 计算 y_{i,t}
            y[t, i] = rho * np.dot(miu_matrix[i, :], y[t - 1, :]) + gamma[t, i]

    # 将 y 矩阵转换为 DataFrame，并添加日期列
    y_df = pd.DataFrame(y, columns=[f'Series_{i + 1}' for i in range(n)])
    dates = pd.date_range(start='2020-01-01', periods=T, freq='D')
    y_df.insert(0, 'Time Index', dates)

    # 保存处理后的时间序列到 Excel 文件
    time_series_file_path = f'E:/BaiduSyncdisk/学习/【研究】/二元统计推断/稿子/3改/结果/密度/{iteration}_data.xlsx'
    y_df.to_excel(time_series_file_path, index=False)
    print(f"Time series saved to {time_series_file_path}")

    # 保存 GARCH 模型参数到 Excel 文件
    garch_params_file_path = f'E:/BaiduSyncdisk/学习/【研究】/二元统计推断/稿子/3改/结果/密度/{iteration}_参.xlsx'
    with pd.ExcelWriter(garch_params_file_path) as writer:
        pd.DataFrame(miu_matrix).to_excel(writer, sheet_name='Mu_Matrix')
        pd.DataFrame(alpha_0, columns=['Alpha_0']).to_excel(writer, sheet_name='Alpha_0')
        pd.DataFrame(alpha_matrix).to_excel(writer, sheet_name='Alpha_Matrix')
        pd.DataFrame(beta_matrix).to_excel(writer, sheet_name='Beta_Matrix')

    print(f"GARCH parameters saved to {garch_params_file_path}")
