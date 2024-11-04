import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

# 读取训练数据
train_data = pd.read_csv('./train/train.csv')

# 分离特征和标签
X = train_data.drop(columns=['ID', 'Diagnosis'])
y = train_data['Diagnosis']

# 划分训练集和验证集（可选）
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 使用KNeighborsClassifier来创建模型并进行训练和预测
knn_classifier = KNeighborsClassifier(n_neighbors=5)

# 训练模型
knn_classifier.fit(X_train, y_train)

# 如果需要，可以在验证集上评估模型
y_pred_val = knn_classifier.predict(X_val)
print(classification_report(y_val, y_pred_val))

# 读取测试数据
test_data = pd.read_csv('./test/test.csv')

# 进行预测
X_test = test_data.drop(columns=['ID'])
test_predictions = knn_classifier.predict(X_test)

# 创建结果 DataFrame
result = pd.DataFrame({
    'ID': test_data['ID'],
    'Diagnosis': test_predictions
})

# 保存结果到 CSV 文件
result.to_csv('result.csv', index=False)
