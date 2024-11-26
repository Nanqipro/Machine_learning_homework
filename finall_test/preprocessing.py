import pandas as pd
from sklearn.model_selection import train_test_split

# 1. 加载数据
data = pd.read_csv('train.csv')  # 请替换为你的数据集路径

# 2. 特征和标签分离
X = data.drop(columns=['Class'])  # 特征列，假设 'Class' 是标签列
y = data['Class']  # 标签列

# 3. 首先将数据集按7:3比例拆分为训练集和临时集（即验证集+测试集）
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)

# 4. 然后将临时集按2:1比例拆分为验证集和测试集
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.333, random_state=42)

# 5. 输出各个数据集的大小
print(f'训练集大小: {X_train.shape[0]}')
print(f'验证集大小: {X_val.shape[0]}')
print(f'测试集大小: {X_test.shape[0]}')

# 6. 如果需要保存为CSV文件
train_df = pd.concat([X_train, y_train], axis=1)
val_df = pd.concat([X_val, y_val], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

train_df.to_csv('./train/train.csv', index=False)
val_df.to_csv('./verify/val.csv', index=False)
test_df.to_csv('./test/test.csv', index=False)

print("数据集已分割并保存为 train.csv, val.csv, test.csv")
