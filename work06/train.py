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

# V1
# import pandas as pd
# from sklearn.semi_supervised import LabelSpreading
# from sklearn.preprocessing import StandardScaler
#
# # 加载数据并提取特征
# train_data = pd.read_csv('train/train.csv')
# test_data = pd.read_csv('test/test.csv')
# X_train, X_test = train_data.iloc[:, :13], test_data.iloc[:, :13]  # 确保 X_test 是 DataFrame
#
# # 创建标签数组
# y_train = [-1] * len(X_train)
# y_train[:3] = [1, 2, 3]
#
# # 标准化
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)
#
# # 半监督学习并预测
# model = LabelSpreading()
# model.fit(X_train_scaled, y_train)
# y_pred = model.predict(X_test_scaled)
#
# # 保存结果
# pd.DataFrame({'Class': y_pred}).to_csv('result.csv', index=False)
# print("结果已保存到result.csv文件中。")


# # V2
# # 导入必要库
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
# from sklearn.preprocessing import StandardScaler, PolynomialFeatures
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score
#
# # 读取数据
# train_data = pd.read_csv('./train/train.csv')
# test_data = pd.read_csv('./test/test.csv')
#
# # 初始化已知标签的数据和无标签的数据
# labeled_indices = [0, 1, 2, 3, 4]
# labeled_labels = [1, 2, 3, 1, 2]
#
# X_known = train_data.iloc[labeled_indices, :]
# y_known = labeled_labels
# X_unknown = train_data.drop(labeled_indices).reset_index(drop=True)
#
# # 特征工程：多项式特征生成和标准化
# poly_features = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
# X_known_poly = poly_features.fit_transform(X_known)
# X_unknown_poly = poly_features.transform(X_unknown)
# X_test_poly = poly_features.transform(test_data)
#
# scaler = StandardScaler()
# X_known_scaled = scaler.fit_transform(X_known_poly)
# X_unknown_scaled = scaler.transform(X_unknown_poly)
# X_test_scaled = scaler.transform(X_test_poly)
#
# # 定义SVM模型和参数网格
# svm_model = SVC(probability=True, random_state=42)
# param_grid = {
#     'C': [0.1, 1.0, 10.0],
#     'kernel': ['linear', 'rbf']
# }
#
# # 模型训练和交叉验证
# print("Training SVM model with cross-validation...")
# grid_search = GridSearchCV(
#     estimator=svm_model,
#     param_grid=param_grid,
#     cv=2,
#     n_jobs=-1,
#     scoring='accuracy'
# )
# grid_search.fit(X_known_scaled, y_known)
# optimal_model = grid_search.best_estimator_
#
# cv_scores = cross_val_score(optimal_model, X_known_scaled, y_known, cv=2, scoring='accuracy')
# mean_score = np.mean(cv_scores)
# std_score = np.std(cv_scores)
#
# print(f"Cross-validation accuracy: {cv_scores}")
# print(f"Mean accuracy: {mean_score:.6f}")
# print(f"Standard deviation: {std_score:.6f}")
#
# # 自训练过程
# def self_train(X_known, y_known, X_unknown, X_test, model, max_iter=10, confidence_threshold=0.9):
#     for i in range(max_iter):
#         # 训练模型
#         model.fit(X_known, y_known)
#
#         # 预测无标签数据
#         y_pred = model.predict(X_unknown)
#         y_prob = model.predict_proba(X_unknown)
#
#         # 选择置信度高的样本
#         confident_samples = [j for j, prob in enumerate(y_prob) if np.max(prob) >= confidence_threshold]
#
#         if not confident_samples:
#             break
#
#         # 将高置信度样本加入已知标签的数据集
#         X_new = X_unknown[confident_samples]
#         y_new = y_pred[confident_samples]
#
#         X_known = np.vstack((X_known, X_new))
#         y_known = np.concatenate([y_known, y_new])
#
#         # 从无标签数据集中移除已标记样本
#         X_unknown = np.delete(X_unknown, confident_samples, axis=0)
#
#     # 最终训练模型并预测测试集
#     model.fit(X_known, y_known)
#     y_test = model.predict(X_test)
#
#     return y_test
#
# # 运行自训练
# y_test_result = self_train(X_known_scaled, y_known, X_unknown_scaled, X_test_scaled, optimal_model)
#
# # 保存结果
# output_df = pd.DataFrame({'Class': y_test_result})
# output_df.to_csv('result.csv', index=False)
#
# # print("Prediction complete. Results saved to 'result.csv'.")


# V3
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 1. 加载数据集
train_data = pd.read_csv('train/train.csv')
test_data = pd.read_csv('test/test.csv')

# 2. 数据预处理 - 特征标准化
scaler = StandardScaler()
scaler.fit(train_data)
train_scaled = scaler.transform(train_data)
test_scaled = scaler.transform(test_data)

# 3. 半监督学习 - KMeans聚类
known_samples = train_scaled[:3]  # 前3个样本作为已知样本
kmeans = KMeans(n_clusters=3, init=known_samples, n_init=1)
kmeans.fit(train_scaled[3:])

# 4. 预测测试数据的类别
test_predictions = kmeans.predict(test_scaled)

# 5. 生成结果文件
result = pd.DataFrame({'Class': test_predictions + 1})
result.to_csv('result.csv', index=False)
