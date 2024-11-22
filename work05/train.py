# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import classification_report
#
# # 读取训练数据
# train_data = pd.read_csv('./train/train.csv')
#
# # 分离特征和标签
# X = train_data.drop(columns=['ID', 'Diagnosis'])
# y = train_data['Diagnosis']
#
# # 划分训练集和验证集（可选）
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # 使用KNeighborsClassifier来创建模型并进行训练和预测
# knn_classifier = KNeighborsClassifier(n_neighbors=5)
#
# # 训练模型
# knn_classifier.fit(X_train, y_train)
#
# # 如果需要，可以在验证集上评估模型
# y_pred_val = knn_classifier.predict(X_val)
# print(classification_report(y_val, y_pred_val))
#
# # 读取测试数据
# test_data = pd.read_csv('./test/test.csv')
#
# # 进行预测
# X_test = test_data.drop(columns=['ID'])
# test_predictions = knn_classifier.predict(X_test)
#
# # 创建结果 DataFrame
# result = pd.DataFrame({
#     'ID': test_data['ID'],
#     'Diagnosis': test_predictions
# })
#
# # 保存结果到 CSV 文件
# result.to_csv('result.csv', index=False)

# V1
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

# 读取训练数据
train_data = pd.read_csv('./train/train.csv')

# 检查缺失值
if train_data.isnull().sum().sum() > 0:
    train_data = train_data.dropna()

# 分离特征和标签
X = train_data.drop(columns=['ID', 'Diagnosis'])
y = train_data['Diagnosis']

# 特征标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# 创建并训练模型（使用默认参数，避免GridSearchCV）
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 在验证集上评估模型
y_pred_val = model.predict(X_val)
print("Classification Report:\n", classification_report(y_val, y_pred_val))

# 读取测试数据
test_data = pd.read_csv('./test/test.csv')

# 检查缺失值
if test_data.isnull().sum().sum() > 0:
    test_data = test_data.dropna()

# 提取特征并标准化
X_test = test_data.drop(columns=['ID'])
X_test_scaled = scaler.transform(X_test)

# 进行预测
test_predictions = model.predict(X_test_scaled)

# 创建结果 DataFrame
result = pd.DataFrame({
    'ID': test_data['ID'],
    'Diagnosis': test_predictions
})

# 保存结果到 CSV 文件
result.to_csv('result.csv', index=False)

# # V2
# import pandas as pd
# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import classification_report
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.utils import resample
#
# # 读取训练数据
# train_data = pd.read_csv('./train/train.csv')
#
# # 检查缺失值
# if train_data.isnull().sum().sum() > 0:
#     train_data = train_data.dropna()
#
# # 分离特征和标签
# X = train_data.drop(columns=['ID', 'Diagnosis'])
# y = train_data['Diagnosis']
#
# # 检查类别分布
# # print("原始类别分布：")
# # print(y.value_counts())
#
# # 处理类别不平衡（上采样少数类）
# df = pd.concat([X, y], axis=1)
# majority = df[df['Diagnosis'] == 0]
# minority = df[df['Diagnosis'] == 1]
#
# minority_upsampled = resample(minority,
#                               replace=True,
#                               n_samples=len(majority),
#                               random_state=42)
#
# df_balanced = pd.concat([majority, minority_upsampled])
#
# X = df_balanced.drop('Diagnosis', axis=1)
# y = df_balanced['Diagnosis']
#
# # 特征标准化
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
#
# # 使用随机森林获取特征重要性
# temp_model = RandomForestClassifier(random_state=42)
# temp_model.fit(X_scaled, y)
# importances = temp_model.feature_importances_
#
# # 创建特征重要性DataFrame
# feature_names = X.columns
# feature_importances = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
# feature_importances = feature_importances.sort_values(by='Importance', ascending=False)
#
# # 选择前15个重要特征
# top_n = 15
# top_features = feature_importances['Feature'].head(top_n).tolist()
# X_top = X[top_features]
#
# # 对选择的特征重新进行标准化
# X_top_scaled = scaler.fit_transform(X_top)
#
# # 划分训练集和验证集
# X_train, X_val, y_train, y_val = train_test_split(
#     X_top_scaled, y, test_size=0.2, random_state=42, stratify=y)
#
# # 手动调整模型参数（随机森林）
# model_rf = RandomForestClassifier(
#     n_estimators=150,
#     max_depth=8,
#     min_samples_split=4,
#     random_state=42
# )
# model_rf.fit(X_train, y_train)
#
# # 在验证集上评估模型（随机森林）
# y_pred_val_rf = model_rf.predict(X_val)
# # print("随机森林分类报告：\n", classification_report(y_val, y_pred_val_rf))
#
# # 使用交叉验证评估模型性能（随机森林）
# scores_rf = cross_val_score(model_rf, X_top_scaled, y, cv=5, scoring='accuracy')
# # print("随机森林交叉验证得分：", scores_rf)
# # print("随机森林平均准确率：", scores_rf.mean())
#
# # 尝试使用梯度提升模型
# model_gb = GradientBoostingClassifier(
#     n_estimators=150,
#     learning_rate=0.1,
#     max_depth=5,
#     random_state=42
# )
# model_gb.fit(X_train, y_train)
#
# # 在验证集上评估模型（梯度提升）
# y_pred_val_gb = model_gb.predict(X_val)
# # print("梯度提升分类报告：\n", classification_report(y_val, y_pred_val_gb))
#
# # 使用交叉验证评估模型性能（梯度提升）
# scores_gb = cross_val_score(model_gb, X_top_scaled, y, cv=5, scoring='accuracy')
# # print("梯度提升交叉验证得分：", scores_gb)
# # print("梯度提升平均准确率：", scores_gb.mean())
#
# # 选择性能更好的模型进行预测（假设梯度提升效果更好）
# best_model = model_gb
#
# # 读取测试数据
# test_data = pd.read_csv('./test/test.csv')
#
# # 检查缺失值
# if test_data.isnull().sum().sum() > 0:
#     test_data = test_data.dropna()
#
# # 提取与训练数据相同的特征
# X_test = test_data[top_features]
#
# # 标准化
# X_test_scaled = scaler.transform(X_test)
#
# # 进行预测
# test_predictions = best_model.predict(X_test_scaled)
#
# # 创建结果 DataFrame
# result = pd.DataFrame({
#     'ID': test_data['ID'],
#     'Diagnosis': test_predictions
# })
#
# # 保存结果到 CSV 文件
# result.to_csv('result.csv', index=False)

# V3
# import pandas as pd
# from sklearn.model_selection import GridSearchCV, StratifiedKFold
# from sklearn.preprocessing import StandardScaler
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import classification_report
# from sklearn.decomposition import PCA
# from sklearn.feature_selection import SelectKBest, f_classif
#
# # 1. 加载数据
# train_data = pd.read_csv('train/train.csv')
# test_data = pd.read_csv('test/test.csv')
#
# # 2. 分离特征和标签
# X_train = train_data.drop(columns=['ID', 'Diagnosis'])
# y_train = train_data['Diagnosis']
#
# X_test = test_data.drop(columns=['ID'])
#
# # 3. 处理缺失值（如果有）
# X_train.fillna(X_train.mean(), inplace=True)
# X_test.fillna(X_test.mean(), inplace=True)
#
# # 4. 特征选择
# # 使用方差分析（ANOVA）选择最佳特征
# selector = SelectKBest(score_func=f_classif, k=20)  # 选择前20个特征
# X_train_selected = selector.fit_transform(X_train, y_train)
# X_test_selected = selector.transform(X_test)
#
# # 5. 特征标准化
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train_selected)
# X_test_scaled = scaler.transform(X_test_selected)
#
# # 6. 降维处理（可选）
# # pca = PCA(n_components=15)
# # X_train_pca = pca.fit_transform(X_train_scaled)
# # X_test_pca = pca.transform(X_test_scaled)
#
# # 7. 定义参数网格
# param_grid = {
#     'n_neighbors': range(1, 21),
#     'weights': ['uniform', 'distance'],
#     'metric': ['euclidean', 'manhattan']
# }
#
# # 8. 配置交叉验证策略
# cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#
# # 9. 使用 GridSearchCV 进行超参数优化
# grid_search = GridSearchCV(
#     estimator=KNeighborsClassifier(),
#     param_grid=param_grid,
#     cv=cv_strategy,
#     scoring='accuracy',
#     n_jobs=-1
# )
# grid_search.fit(X_train_scaled, y_train)
#
# # 输出最佳参数
# best_params = grid_search.best_params_
# print(f"最佳参数：{best_params}")
#
# # 输出最佳模型的交叉验证得分
# best_score = grid_search.best_score_
# # print(f"最佳交叉验证准确率：{best_score:.4f}")
#
# # 10. 使用最佳参数训练模型
# best_knn = grid_search.best_estimator_
# best_knn.fit(X_train_scaled, y_train)
#
# # 11. 在训练集上评估模型
# y_train_pred = best_knn.predict(X_train_scaled)
# # print("训练集分类报告：")
# print(classification_report(y_train, y_train_pred))
#
# # 12. 预测测试集
# test_predictions = best_knn.predict(X_test_scaled)
#
# # 13. 保存结果
# result_dataframe = pd.DataFrame({
#     'ID': test_data['ID'],
#     'Diagnosis': test_predictions
# })
# result_dataframe.to_csv('result.csv', index=False)
#
# # print("预测结果已成功保存到 'result.csv'。")

