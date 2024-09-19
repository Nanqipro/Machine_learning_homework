import pandas as pd
from sklearn.linear_model import LinearRegression

# 读取训练数据
train_data = pd.read_csv('train/train.csv')

# 分割特征和目标变量
X_train = train_data.iloc[:, :-1]
y_train = train_data.iloc[:, -1]

# 读取测试数据
test_data = pd.read_csv('test/test.csv')
X_test = test_data

# 创建并训练线性回归模型
model = LinearRegression()
model.fit(X_train, y_train)

# 对测试数据进行预测
y_pred = model.predict(X_test)

# 将预测结果保存到result.csv
output = pd.DataFrame({'age': y_pred})
output.to_csv('result.csv', index=False)
