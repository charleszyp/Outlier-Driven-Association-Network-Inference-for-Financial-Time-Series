import numpy as np
import pandas as pd

def calculate_Pi_j(S, nod_index):
    """
    计算 P_i^j，对于所有 i ≠ j
    """
    m, n = S.shape
    pid = np.delete(np.arange(n), nod_index)  # 节点编号，排除节点 j
    P_i_j = np.zeros(n - 1)

    for idx, i in enumerate(pid):
        # 找到满足条件的时间索引
        indices = np.where((S[:-1, nod_index] == 0) & (S[:-1, i] == 1))[0]
        numerator = np.sum(S[indices + 1, nod_index] == 1)
        denominator = len(indices)
        if denominator > 0:
            P_i_j[idx] = numerator / denominator
        else:
            P_i_j[idx] = 0  # 如果分母为零，概率设为零
    return pid, P_i_j

def EMSCR(S, nod):
    """
    EM 算法进行网络重构
    输入:
        S: 时间序列矩阵
        nod: 要重建的节点编号（从 1 开始）
    输出:
        p: 与其他节点的连接概率
        k: 迭代次数
    """
    m, n = S.shape
    nod_index = nod - 1  # 调整索引，从 0 开始

    # 初始化 P_{i→j} 和 ε_j
    pid, P_i_j = calculate_Pi_j(S, nod_index)
    P_ij = np.full(len(pid), 0.5)  # 初始值为 0.5
    epsilon_j = 0.5  # 初始值为 0.5

    # 获取所有满足 Ψ_j(t_m) = 0 的时间索引
    indices_m = np.where(S[:-1, nod_index] == 0)[0]
    M = len(indices_m)
    if M == 0:
        # 如果没有满足条件的数据，返回空结果
        return None, 0

    # 设置收敛条件
    tol = 1e-4
    max_iter = 1000
    delta = np.inf
    k = 0

    while delta > tol and k < max_iter:
        k += 1
        P_ij_old = P_ij.copy()
        epsilon_j_old = epsilon_j

        # E 步骤
        rho_i = np.zeros((M, len(pid)))
        rho_epsilon = np.zeros(M)

        for idx_m, m in enumerate(indices_m):
            # 计算 E_j^{(t_m+1)}
            E_j_tm1 = np.sum(P_ij * P_i_j * S[m, pid]) + epsilon_j

            # 防止除以零
            if E_j_tm1 == 0:
                continue

            # 计算 rho_i^(t_m) 和 rho_{ε_j}^{(t_m)}
            rho_i[idx_m, :] = (P_ij * P_i_j * S[m, pid]) / E_j_tm1
            rho_epsilon[idx_m] = epsilon_j / E_j_tm1

        # M 步骤

        # 更新 P_{i→j}
        numerator = np.sum((S[indices_m + 1, nod_index][:, np.newaxis] * rho_i), axis=0)
        denominator = np.sum(P_i_j * S[indices_m][:, pid], axis=0)
        P_ij = numerator / (denominator + 1e-7)  # 防止除以零

        # 更新 ε_j
        numerator_epsilon = np.sum(S[indices_m + 1, nod_index] * rho_epsilon)
        epsilon_j = numerator_epsilon / M

        # 防止数值问题
        P_ij = np.clip(P_ij, 0, 1)
        epsilon_j = max(min(epsilon_j, 1), 0)

        # 计算变化量
        delta_Pij = np.sum(np.abs(P_ij - P_ij_old))
        delta_epsilon = np.abs(epsilon_j - epsilon_j_old)
        delta = delta_Pij + delta_epsilon

    # 准备输出
    p = np.vstack((pid + 1, P_ij))  # 调整索引为从 1 开始
    return p, k

def clc_w(S):
    """
    计算两体结果
    输入:
        S: 时间序列矩阵
    输出:
        res_w: 两体结果矩阵
    """
    m, n = S.shape
    res_w = np.zeros((n, n))  # 保存两体计算结果

    for nod in range(1, n + 1):
        result = EMSCR(S, nod)
        if result[0] is not None:
            p, k = result
            nod_index = nod - 1
            res_w[nod_index, p[0, :].astype(int) - 1] = p[1, :]
        else:
            print(f'节点 {nod} 无法进行重构，可能缺少足够的数据。')

    return res_w

# 主程序
for fileNum in range(1, 101):
    filename = f'C:/Users/charles/Desktop/{fileNum}-01.xlsx'
    data = pd.read_excel(filename)
    state_nodes = data.iloc[:, 1:].to_numpy()  # 假设第一列是索引，跳过它
    T = state_nodes.shape[0]

    ave_number = 1

    res1 = []  # 保存两体结果

    for i in range(ave_number):
        print(f'Processing average number: {i+1} out of {ave_number}')
        SS = state_nodes
        print(f'  Processing time series length: {T} out of {T}')
        res_w = clc_w(SS)
        res1.append(res_w)

    # 将 res_w 保存为新的 Excel 文件
    outputFilename = f'C:/Users/charles/Desktop/{fileNum}-统计.xlsx'
    res_w_df = pd.DataFrame(res_w, columns=[f'Var{i+1}' for i in range(res_w.shape[1])],
                            index=[f'Row{i+1}' for i in range(res_w.shape[0])])
    res_w_df.to_excel(outputFilename)
