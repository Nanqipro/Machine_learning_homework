# import pandas as pd
# from sklearn.linear_model import LinearRegression
#
# # 读取训练数据
# train_data = pd.read_csv('train/train.csv')
#
# # 分割特征和目标变量
# X_train = train_data.iloc[:, :-1]
# y_train = train_data.iloc[:, -1]
#
# # 读取测试数据
# test_data = pd.read_csv('test/test.csv')
# X_test = test_data
#
# # 创建并训练线性回归模型
# model = LinearRegression()
# model.fit(X_train, y_train)
#
# # 对测试数据进行预测
# y_pred = model.predict(X_test)
#
# # 将预测结果保存到result.csv
# output = pd.DataFrame({'age': y_pred})
# output.to_csv('result.csv', index=False)

# V1 优化
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

# 读取训练数据
train_data = pd.read_csv('train/train.csv')

# 分割特征和目标变量
X_train = train_data.iloc[:, :-1]
y_train = train_data.iloc[:, -1]

# 读取测试数据
X_test = pd.read_csv('test/test.csv')

# 创建数据处理和模型训练的管道
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RidgeCV(alphas=[0.1, 1.0, 10.0], cv=5))
])

# 训练模型
pipeline.fit(X_train, y_train)

# 评估模型性能（可选）
scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
# print(f"平均交叉验证得分（MSE）：{-scores.mean()}")

# 对测试数据进行预测
y_pred = pipeline.predict(X_test)

# 将预测结果保存到result.csv
output = pd.DataFrame({'age': y_pred})
output.to_csv('result.csv', index=False)
