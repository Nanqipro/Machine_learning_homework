import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 读取训练和测试数据
train_data = pd.read_csv('train/train.csv')
test_data = pd.read_csv('test/test.csv')

# 标准化数据
scaler = StandardScaler()
train_data_scaled = scaler.fit_transform(train_data)
test_data_scaled = scaler.transform(test_data)

# 初始化已知标签
initial_labels = [1, 2, 3] + [None] * (len(train_data) - 3)
train_labels = pd.Series(initial_labels)

# 创建并训练聚类模型
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(train_data_scaled)

# 获取聚类结果
train_data['cluster'] = kmeans.labels_

# 使用模型预测测试数据类别
test_predictions = kmeans.predict(test_data_scaled)

# 将预测结果输出为result.csv，注意列名改为 "Class"
result = pd.DataFrame(test_predictions, columns=['Class'])
result['Class'] = result['Class'].map(lambda x: x + 1)  # 将0,1,2映射到1,2,3
result.to_csv('result.csv', index=False)
