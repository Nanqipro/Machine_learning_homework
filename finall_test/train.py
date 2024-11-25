# import pandas as pd
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from imblearn.over_sampling import SMOTE
# from sklearn.metrics import classification_report
# from sklearn.utils.class_weight import compute_class_weight
#
# # 1. 数据加载
# train_data = pd.read_csv('train/train.csv')
# test_data = pd.read_csv('test/test.csv')
#
# # 2. 数据预处理
# X_train = train_data.drop(columns=['Class'])  # 特征
# y_train = train_data['Class']  # 标签
#
# # 标准化特征
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
#
# # 处理类别不平衡问题（SMOTE过采样）
# smote = SMOTE(random_state=42)
# X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
#
# # 3. 模型训练
# # 使用RandomForestClassifier（也可以尝试其他模型）
# model = RandomForestClassifier(random_state=42, class_weight='balanced')
# model.fit(X_train_resampled, y_train_resampled)
#
# # 4. 对测试集进行预测
# X_test = test_data  # 测试集没有Class列，只需要特征
# X_test_scaled = scaler.transform(X_test)  # 标准化测试集特征
#
# # 预测结果
# y_pred = model.predict(X_test_scaled)
#
# # 5. 输出预测结果到 result.csv
# result = pd.DataFrame(y_pred, columns=['Class'])
# result.to_csv('result.csv', index=False)
#
# print("预测结果已保存到 'result.csv'")

# XGBOOST
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import numpy as np

# 1. 加载数据
train_df = pd.read_csv('train/train.csv')
test_df = pd.read_csv('test/test.csv')

# 2. 特征和标签分离
X_train = train_df.drop(columns=['Class'])
y_train = train_df['Class']
X_test = test_df  # 测试集没有Class列，直接用特征进行预测

# 3. 处理不平衡数据：使用SMOTE进行过采样
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# 4. 标准化处理
scaler = StandardScaler()
X_train_res = scaler.fit_transform(X_train_res)
X_test = scaler.transform(X_test)

# 5. XGBoost模型训练
# 创建DMatrix，这是XGBoost的输入格式
dtrain = xgb.DMatrix(X_train_res, label=y_train_res)
dtest = xgb.DMatrix(X_test)

# 设置XGBoost参数
params = {
    'objective': 'binary:logistic',   # 二分类任务
    'eval_metric': 'logloss',          # 评估指标为对数损失
    'scale_pos_weight': len(y_train_res) / np.sum(y_train_res),  # 处理类别不平衡
    'max_depth': 6,                    # 树的最大深度
    'learning_rate': 0.1               # 学习率
}

# 训练模型
num_round = 100  # 迭代次数，等同于树的数量
model = xgb.train(params, dtrain, num_round)

# 6. 在测试集上进行预测
y_test_pred = model.predict(dtest)
y_test_pred = (y_test_pred > 0.5).astype(int)  # 转换为0或1

# 7. 保存预测结果
result_df = pd.DataFrame(y_test_pred, columns=['Class'])
result_df.to_csv('result.csv', index=False)

print("预测结果已保存至 result.csv")

# # 优化的XGBOOST01
# import pandas as pd
# import xgboost as xgb
# from sklearn.preprocessing import StandardScaler
# from imblearn.over_sampling import SMOTE
# from sklearn.model_selection import train_test_split
# import numpy as np
#
#
# # 1. 加载数据
# def load_data(train_path, test_path):
#     train_df = pd.read_csv(train_path)
#     test_df = pd.read_csv(test_path)
#     return train_df, test_df
#
#
# # 2. 特征和标签分离
# def split_features_labels(train_df):
#     X_train = train_df.drop(columns=['Class'])
#     y_train = train_df['Class']
#     return X_train, y_train
#
#
# # 3. 处理不平衡数据：SMOTE
# def apply_smote(X_train, y_train):
#     smote = SMOTE(random_state=42)
#     X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
#     return X_train_res, y_train_res
#
#
# # 4. 数据标准化
# def standardize_data(X_train_res, X_test):
#     scaler = StandardScaler()
#     X_train_res = scaler.fit_transform(X_train_res)
#     X_test = scaler.transform(X_test)
#     return X_train_res, X_test
#
#
# # 5. 创建XGBoost模型
# def train_xgboost(X_train_res, y_train_res, X_test):
#     # 设置XGBoost参数
#     params = {
#         'objective': 'binary:logistic',
#         'eval_metric': 'logloss',
#         'scale_pos_weight': len(y_train_res) / np.sum(y_train_res),
#         'max_depth': 6,
#         'learning_rate': 0.1,
#         'subsample': 0.8,
#         'colsample_bytree': 0.8,
#     }
#
#     # 训练模型
#     dtrain = xgb.DMatrix(X_train_res, label=y_train_res)
#     dtest = xgb.DMatrix(X_test)
#
#     num_round = 100
#     model = xgb.train(params, dtrain, num_round)
#
#     # 进行预测
#     y_test_pred = model.predict(dtest)
#     y_test_pred = (y_test_pred > 0.5).astype(int)
#
#     return y_test_pred
#
#
# # 6. 保存预测结果
# def save_results(y_test_pred, output_path):
#     result_df = pd.DataFrame(y_test_pred, columns=['Class'])
#     result_df.to_csv(output_path, index=False)
#     print(f"预测结果已保存至 {output_path}")
#
#
# # 主函数
# def main():
#     train_path = 'train/train.csv'
#     test_path = 'test/test.csv'
#     output_path = 'result.csv'
#
#     # 加载数据
#     train_df, test_df = load_data(train_path, test_path)
#
#     # 特征和标签分离
#     X_train, y_train = split_features_labels(train_df)
#     X_test = test_df  # 测试集没有Class列，直接使用特征
#
#     # 处理不平衡数据
#     X_train_res, y_train_res = apply_smote(X_train, y_train)
#
#     # 数据标准化
#     X_train_res, X_test = standardize_data(X_train_res, X_test)
#
#     # 训练模型并预测
#     y_test_pred = train_xgboost(X_train_res, y_train_res, X_test)
#
#     # 保存预测结果
#     save_results(y_test_pred, output_path)
#
#
# if __name__ == "__main__":
#     main()

# # 优化的XGBOOST02
# import pandas as pd
# import xgboost as xgb
# import lightgbm as lgb
# from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
# from sklearn.preprocessing import StandardScaler
# from imblearn.over_sampling import SMOTE
# from sklearn.metrics import f1_score, roc_auc_score, classification_report
# import numpy as np
#
#
# # 1. 加载数据
# def load_data(train_path, test_path):
#     train_df = pd.read_csv(train_path)
#     test_df = pd.read_csv(test_path)
#     return train_df, test_df
#
#
# # 2. 特征和标签分离
# def split_features_labels(train_df):
#     X_train = train_df.drop(columns=['Class'])
#     y_train = train_df['Class']
#     return X_train, y_train
#
#
# # 3. 处理不平衡数据：SMOTE
# def apply_smote(X_train, y_train):
#     smote = SMOTE(random_state=42, sampling_strategy='auto')  # 可调sampling_strategy参数
#     X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
#     return X_train_res, y_train_res
#
#
# # 4. 数据标准化
# def standardize_data(X_train_res, X_test):
#     scaler = StandardScaler()
#     X_train_res = scaler.fit_transform(X_train_res)
#     X_test = scaler.transform(X_test)
#     return X_train_res, X_test
#
#
# # 5. 创建XGBoost模型
# def train_xgboost(X_train_res, y_train_res, X_test):
#     # 设置XGBoost参数
#     params = {
#         'objective': 'binary:logistic',
#         'eval_metric': 'logloss',
#         'scale_pos_weight': len(y_train_res) / np.sum(y_train_res),
#         'max_depth': 6,
#         'learning_rate': 0.1,
#         'subsample': 0.8,
#         'colsample_bytree': 0.8,
#     }
#
#     # 训练模型
#     dtrain = xgb.DMatrix(X_train_res, label=y_train_res)
#     dtest = xgb.DMatrix(X_test)
#
#     num_round = 100
#     model = xgb.train(params, dtrain, num_round)
#
#     # 进行预测
#     y_test_pred = model.predict(dtest)
#     y_test_pred = (y_test_pred > 0.5).astype(int)  # 使用阈值0.5，确保输出为0或1
#
#     return y_test_pred, model
#
#
# # 6. LightGBM模型
# def train_lightgbm(X_train_res, y_train_res, X_test):
#     params = {
#         'objective': 'binary',
#         'metric': 'binary_error',
#         'boosting_type': 'gbdt',
#         'num_leaves': 31,
#         'learning_rate': 0.05,
#         'feature_fraction': 0.9,
#         'bagging_fraction': 0.8,
#         'bagging_freq': 5,
#         'scale_pos_weight': len(y_train_res) / np.sum(y_train_res),
#     }
#
#     # 训练模型
#     train_data = lgb.Dataset(X_train_res, label=y_train_res)
#     model = lgb.train(params, train_data, num_boost_round=100)
#
#     # 预测
#     y_test_pred = model.predict(X_test)
#     y_test_pred = (y_test_pred > 0.5).astype(int)  # 使用阈值0.5，确保输出为0或1
#
#     return y_test_pred, model
#
#
# # 7. 集成学习：XGBoost 和 LightGBM 融合
# def ensemble_predictions(xgb_preds, lgb_preds):
#     return np.round((xgb_preds + lgb_preds) / 2).astype(int)  # 确保输出为整数类型
#
#
# # 8. 评估模型性能
# def evaluate_model(y_true, y_pred):
#     f1 = f1_score(y_true, y_pred)
#     auc = roc_auc_score(y_true, y_pred)
#     print(f"F1-Score: {f1}")
#     print(f"ROC-AUC: {auc}")
#     print(classification_report(y_true, y_pred))
#
#
# # 9. 保存预测结果
# def save_results(y_test_pred, output_path):
#     result_df = pd.DataFrame(y_test_pred, columns=['Class'])
#     result_df.to_csv(output_path, index=False)
#     print(f"预测结果已保存至 {output_path}")
#
#
# # 主函数
# def main():
#     train_path = 'train/train.csv'
#     test_path = 'test/test.csv'
#     output_path = 'result.csv'
#
#     # 加载数据
#     train_df, test_df = load_data(train_path, test_path)
#
#     # 特征和标签分离
#     X_train, y_train = split_features_labels(train_df)
#     X_test = test_df  # 测试集没有Class列，直接使用特征
#
#     # 处理不平衡数据
#     X_train_res, y_train_res = apply_smote(X_train, y_train)
#
#     # 数据标准化
#     X_train_res, X_test = standardize_data(X_train_res, X_test)
#
#     # 训练XGBoost模型并预测
#     xgb_preds, xgb_model = train_xgboost(X_train_res, y_train_res, X_test)
#
#     # 训练LightGBM模型并预测
#     lgb_preds, lgb_model = train_lightgbm(X_train_res, y_train_res, X_test)
#
#     # 集成预测结果
#     y_test_pred = ensemble_predictions(xgb_preds, lgb_preds)
#
#     # 评估模型性能（使用真实标签进行评估）
#     # 如果有真实标签的话，可以做评估
#     # evaluate_model(y_test, y_test_pred)
#
#     # 保存预测结果
#     save_results(y_test_pred, output_path)
#
#
# if __name__ == "__main__":
#     main()
#


# # LightGBM
# import pandas as pd
# import lightgbm as lgb
# from sklearn.preprocessing import StandardScaler
# from imblearn.over_sampling import SMOTE
# import numpy as np
#
# # 1. 加载数据
# train_df = pd.read_csv('train/train.csv')
# test_df = pd.read_csv('test/test.csv')
#
# # 2. 特征和标签分离
# X_train = train_df.drop(columns=['Class'])
# y_train = train_df['Class']
# X_test = test_df  # 测试集没有Class列，直接用特征进行预测
#
# # 3. 处理不平衡数据：使用SMOTE进行过采样
# smote = SMOTE(random_state=42)
# X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
#
# # 4. 标准化处理
# scaler = StandardScaler()
# X_train_res = scaler.fit_transform(X_train_res)
# X_test = scaler.transform(X_test)
#
# # 5. LightGBM模型训练
# # 创建数据集
# train_data = lgb.Dataset(X_train_res, label=y_train_res)
# test_data = lgb.Dataset(X_test, reference=train_data)
#
# # 设置LightGBM的参数
# params = {
#     'objective': 'binary',          # 二分类任务
#     'metric': 'binary_logloss',     # 评估指标为对数损失
#     'is_unbalance': True,           # 处理类别不平衡
#     'boosting_type': 'gbdt',        # 使用梯度提升树（GBDT）
#     'num_leaves': 31,               # 树的最大叶子数
#     'learning_rate': 0.05,          # 学习率
#     'feature_fraction': 0.9,        # 特征选择的比例
#     'bagging_fraction': 0.8,        # 数据选择的比例
#     'bagging_freq': 5,              # 每5次迭代进行一次bagging
#     'num_threads': 4                # 线程数
# }
#
# # 训练模型
# num_round = 100  # 迭代次数
# # 将early_stopping_rounds传入训练参数，确保valid_sets包含验证集
# model = lgb.train(
#     params,
#     train_data,
#     num_round,
#     valid_sets=[train_data, test_data],  # 同时使用训练集和测试集作为验证集
#     early_stopping_rounds=10  # 设置提前停止
# )
#
# # 6. 在测试集上进行预测
# y_test_pred = model.predict(X_test, num_iteration=model.best_iteration)
# y_test_pred = (y_test_pred > 0.5).astype(int)  # 转换为0或1
#
# # 7. 保存预测结果
# result_df = pd.DataFrame(y_test_pred, columns=['class'])
# result_df.to_csv('result.csv', index=False)
#
# print("预测结果已保存至 result.csv")

# # GBM
# import pandas as pd
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.preprocessing import StandardScaler
# from imblearn.over_sampling import SMOTE
# from sklearn.metrics import classification_report
# import numpy as np
#
# # 1. 加载数据
# train_df = pd.read_csv('train/train.csv')
# test_df = pd.read_csv('test/test.csv')
#
# # 2. 特征和标签分离
# X_train = train_df.drop(columns=['Class'])
# y_train = train_df['Class']
# X_test = test_df  # 测试集没有Class列，直接用特征进行预测
#
# # 3. 处理不平衡数据：使用SMOTE进行过采样
# smote = SMOTE(random_state=42)
# X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
#
# # 4. 标准化处理
# scaler = StandardScaler()
# X_train_res = scaler.fit_transform(X_train_res)
# X_test = scaler.transform(X_test)
#
# # 5. 梯度提升机（GBM）模型训练
# gbm_model = GradientBoostingClassifier(
#     n_estimators=100,  # 树的数量
#     learning_rate=0.1, # 学习率
#     max_depth=6,       # 树的最大深度
#     subsample=0.8,     # 数据采样比例
#     random_state=42    # 固定随机种子
# )
#
# # 训练模型
# gbm_model.fit(X_train_res, y_train_res)
#
# # 6. 在测试集上进行预测
# y_test_pred = gbm_model.predict(X_test)
#
# # 7. 保存预测结果
# result_df = pd.DataFrame(y_test_pred, columns=['class'])
# result_df.to_csv('result.csv', index=False)
#
# print("预测结果已保存至 result.csv")
