# 导入所需的库
import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# 加载训练数据
train_data = pd.read_csv('./train/train.csv')

# 加载测试数据
test_data = pd.read_csv('./test/test.csv')

# 处理缺失值
train_data = train_data.dropna()
test_data = test_data.dropna()

# 标签编码将分类数据转化为数值
label_encoders = {}
categorical_columns = ['workplace', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country']

for col in categorical_columns:
    label_encoders[col] = LabelEncoder()
    train_data[col] = label_encoders[col].fit_transform(train_data[col])

    # 在测试集中遇到未见过的类别时，将其填充为训练集中最常见的类别
    test_data[col] = test_data[col].apply(lambda x: x if x in label_encoders[col].classes_ else 'Unknown')
    # 扩展label encoder的类别
    label_encoders[col].classes_ = np.append(label_encoders[col].classes_, 'Unknown')
    test_data[col] = label_encoders[col].transform(test_data[col])

# 分离特征和目标变量
X = train_data.drop(columns=['income'])
y = train_data['income']

# 分割数据为训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# XGBoost模型
xgb_model = xgb.XGBClassifier()

# 训练模型
xgb_model.fit(X_train, y_train)

# 在验证集上预测
y_pred = xgb_model.predict(X_val)

# 计算准确率
accuracy = accuracy_score(y_val, y_pred)
print(f'Accuracy on validation set: {accuracy}')

# 对测试数据进行预测
test_predictions = xgb_model.predict(test_data)

# 将预测结果保存到CSV文件
output = pd.DataFrame({'income': test_predictions})
output.to_csv('result.csv', index=False)

print("Predictions saved to result.csv.")
