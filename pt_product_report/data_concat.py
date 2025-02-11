import os
import pandas as pd

# 获取文件夹中所有 Excel 文件
folder_path = r"C:\Users\Administrator\Desktop\新建文件夹\sellersprite-export-table(25)-20250122"
files = [f for f in os.listdir(folder_path) if f.endswith('.xlsx') or f.endswith('.xls')]

# 存储所有合并的数据
combined_data = []

for file in files:
    file_path = os.path.join(folder_path, file)

    # 读取 Excel 文件的第一个 sheet（不包含首行）
    df = pd.read_excel(file_path, sheet_name=0, header=1)  # header=1 跳过首行

    # 添加文件名作为首列
    df.insert(0, 'File Name', file)

    # 将数据加入合并数据列表
    combined_data.append(df)

# 合并所有数据
final_df = pd.concat(combined_data, ignore_index=True)

# 保存到一个新的 Excel 文件
final_df.to_excel(os.path.join(folder_path, r"C:\Users\Administrator\Desktop\新建文件夹\Hossmily_Ziimaikery.xlsx"), index=False)

print("合并完成！")
