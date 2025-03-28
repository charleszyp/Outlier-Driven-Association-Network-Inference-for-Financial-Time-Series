import numpy as np
import pandas as pd
from tqdm import tqdm  # 导入 tqdm

def calculate_Pi_j(S, nod_index, lag_n):
    """
    计算 P_i^j，对于所有 i ≠ j，考虑滞后时间 n
    """
    T, N = S.shape
    pid = np.delete(np.arange(N), nod_index)  # 节点编号，排除节点 j
    P_i_j = np.zeros(N - 1)

    for idx, i in enumerate(pid):
        indices = np.where((S[:T - lag_n, nod_index] == 0) & (S[:T - lag_n, i] == 1))[0]
        numerator = np.sum(S[indices + lag_n, nod_index] == 1)
        denominator = len(indices)
        P_i_j[idx] = numerator / denominator if denominator > 0 else 0

    return pid, P_i_j

def EMSCR(S, nod, lag_n):
    """
    EM 算法进行网络重构，考虑滞后时间 n
    """
    T, N = S.shape
    nod_index = nod - 1

    pid, P_i_j = calculate_Pi_j(S, nod_index, lag_n)
    P_ij = np.full(len(pid), 0.5)
    epsilon_j = 0.5

    indices_m = np.where(S[:T - lag_n, nod_index] == 0)[0]
    M = len(indices_m)
    if M == 0:
        return None, 0

    tol, max_iter, delta = 1e-4, 1000, np.inf
    k = 0

    while delta > tol and k < max_iter:
        k += 1
        P_ij_old, epsilon_j_old = P_ij.copy(), epsilon_j

        rho_i = np.zeros((M, len(pid)))
        rho_epsilon = np.zeros(M)

        for idx_m, m in enumerate(indices_m):
            E_j_tm_n = np.sum(P_ij * P_i_j * S[m, pid]) + epsilon_j
            if E_j_tm_n == 0:
                continue

            rho_i[idx_m, :] = (P_ij * P_i_j * S[m, pid]) / E_j_tm_n
            rho_epsilon[idx_m] = epsilon_j / E_j_tm_n

        numerator = np.sum((S[indices_m + lag_n, nod_index][:, np.newaxis] * rho_i), axis=0)
        denominator = np.sum(P_i_j * S[indices_m][:, pid], axis=0)
        P_ij = numerator / (denominator + 1e-7)

        numerator_epsilon = np.sum(S[indices_m + lag_n, nod_index] * rho_epsilon)
        epsilon_j = numerator_epsilon / M

        P_ij = np.clip(P_ij, 0, 1)
        epsilon_j = max(min(epsilon_j, 1), 0)

        delta = np.sum(np.abs(P_ij - P_ij_old)) + np.abs(epsilon_j - epsilon_j_old)

    p = np.vstack((pid + 1, P_ij))
    return p, k

def clc_w(S, lag_n):
    """
    计算两体结果，考虑滞后时间 n
    """
    T, N = S.shape
    res_w = np.zeros((N, N))

    for nod in tqdm(range(1, N + 1), desc="Processing nodes"):
        result = EMSCR(S, nod, lag_n)
        if result[0] is not None:
            p, k = result
            nod_index = nod - 1
            res_w[nod_index, p[0, :].astype(int) - 1] = p[1, :]
        else:
            print(f'节点 {nod} 无法进行重构，可能缺少足够的数据。')

    return res_w

# 主程序
n_values = [1, 2, 3, 4, 5]

for fileNum in tqdm(range(1, 11), desc="Processing files"):
    filename = f'E:/BaiduSyncdisk/学习/【研究】/二元统计推断/稿子/3改/结果/密度/{fileNum}-01.xlsx'
    data = pd.read_excel(filename)
    state_nodes = data.iloc[:, 1:].to_numpy()

    for n in tqdm(n_values, desc=f"Processing lag values for file {fileNum}", leave=False):
        res_w = clc_w(state_nodes, n)
        outputFilename = f'E:/BaiduSyncdisk/学习/【研究】/二元统计推断/稿子/3改/结果/密度/{fileNum}-统计_n{n}.xlsx'
        res_w_df = pd.DataFrame(res_w, columns=[f'Node{i+1}' for i in range(res_w.shape[1])],
                                index=[f'Node{i+1}' for i in range(res_w.shape[0])])
        res_w_df.to_excel(outputFilename)
