import pandas as pd

# 设置循环次数（假设要处理10个文件）
for iteration in range(1, 11):
    # 动态设置文件路径
    file_path = f'E:/BaiduSyncdisk/学习/【研究】/二元统计推断/稿子/3改/结果/密度/{iteration}_data.xlsx'
    data = pd.read_excel(file_path)

    # 初始化一个新的DataFrame来存储时间索引和异常值标记
    result = pd.DataFrame()
    result['Time Index'] = data['Time Index']  # 假设时间索引列名为 'Time Index'

    # 设置异常值识别的灵敏度
    multiplier = 1.5  # 可以调整这个值，比如1.0, 1.5, 2.0等

    # 对每个数值列应用箱形图异常值标记
    for column in data.select_dtypes(include=['number']).columns:
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR

        # 标记异常值并加入结果DataFrame
        result[column + '_outlier'] = ((data[column] < lower_bound) | (data[column] > upper_bound)).astype(int)

    # 动态设置输出路径
    output_path = f'E:/BaiduSyncdisk/学习/【研究】/二元统计推断/稿子/3改/结果/密度/{iteration}-01.xlsx'
    result.to_excel(output_path, index=False)

    print(f"Iteration {iteration}: 处理完成，标记的数据已保存到 {output_path}")
