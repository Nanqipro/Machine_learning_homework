import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 读取训练数据
train_data = pd.read_csv('train/train.csv')

# 提取特征和标签
X_train = train_data[['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']]
y_train = train_data['Species']

# 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# 创建KNN模型
knn = KNeighborsClassifier(n_neighbors=3)

# 训练模型
knn.fit(X_train_scaled, y_train)

# 读取测试数据
test_data = pd.read_csv('test/test.csv')
X_test = test_data[['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']]
X_test_scaled = scaler.transform(X_test)

# 进行预测
predictions = knn.predict(X_test_scaled)

# 生成结果文件
result = pd.DataFrame({
    'Id': test_data['Id'],
    'Species': predictions
})

# 保存结果为CSV
result.to_csv('result.csv', index=False)
