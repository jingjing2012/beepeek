import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 替换为你系统中可用的中文字体
plt.rcParams['axes.unicode_minus'] = False

# 创建示例数据
data = np.random.rand(10, 12)
df = pd.DataFrame(data, columns=[f'列{j}' for j in range(1, 13)], index=[f'行{i}' for i in range(1, 11)])

# 计算相关性矩阵
corr = df.corr()

# 创建热图
sns.heatmap(corr, annot=True, fmt='.2f', linewidths=0.5, linecolor='gray', cmap='coolwarm')

# 设置轴标签为中文
plt.xlabel('列')
plt.ylabel('行')
plt.title('示例相关性热图')

# 显示图形
plt.show()
