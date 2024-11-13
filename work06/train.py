# import pandas as pd
# from sklearn.semi_supervised import LabelSpreading
# from sklearn.preprocessing import StandardScaler
#
# # 加载数据
# train_data = pd.read_csv('train/train.csv')
# test_data = pd.read_csv('test/test.csv')
#
# # 提取特征
# X_train = train_data.iloc[:, :13].values  # 取前13列为特征
# X_test = test_data.values  # 测试数据的13个特征
#
# # 创建标签数组，前3个样本已知类别，其余样本标记为 -1
# y_train = [-1] * len(X_train)
# y_train[0] = 1  # 第一个样本类别为1
# y_train[1] = 2  # 第二个样本类别为2
# y_train[2] = 3  # 第三个样本类别为3
#
# # 数据标准化
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)
#
# # 使用Label Spreading模型进行半监督学习
# model = LabelSpreading()
# model.fit(X_train_scaled, y_train)
#
# # 预测测试数据类别
# y_pred = model.predict(X_test_scaled)
#
# # 将预测结果保存至result.csv
# result = pd.DataFrame({
#     'Class': y_pred
# })
# result.to_csv('result.csv', index=False)
# print("结果已保存到result.csv文件中。")


import pandas as pd
from sklearn.semi_supervised import LabelSpreading
from sklearn.preprocessing import StandardScaler

# 加载数据并提取特征
train_data = pd.read_csv('train/train.csv')
test_data = pd.read_csv('test/test.csv')
X_train, X_test = train_data.iloc[:, :13], test_data.iloc[:, :13]  # 确保 X_test 是 DataFrame

# 创建标签数组
y_train = [-1] * len(X_train)
y_train[:3] = [1, 2, 3]

# 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 半监督学习并预测
model = LabelSpreading()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# 保存结果
pd.DataFrame({'Class': y_pred}).to_csv('result.csv', index=False)
print("结果已保存到result.csv文件中。")



