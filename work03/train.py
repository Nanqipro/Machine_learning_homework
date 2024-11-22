# import os
# import numpy as np
# import sys
# import pandas as pd
# from sklearn import svm
# from sklearn.preprocessing import StandardScaler
#
# def load_data(data_dir):
#     features = []
#     labels = []
#     for filename in os.listdir(data_dir):
#         if filename.endswith('.txt'):
#             # 提取标签
#             label = int(filename.split('_')[0])
#             # 读取文件内容并转换为特征向量
#             with open(os.path.join(data_dir, filename), 'r') as file:
#                 matrix = file.readlines()
#                 vector = np.array([int(pixel) for row in matrix for pixel in row.strip()])
#                 features.append(vector)
#                 labels.append(label)
#     return np.array(features), np.array(labels)
#
# def main():
#     # 加载训练数据
#     X_train, y_train = load_data('train/')
#
#     # 数据标准化
#     scaler = StandardScaler()
#     X_train = scaler.fit_transform(X_train)
#
#     # 训练 SVM 模型
#     model = svm.SVC(gamma='scale')
#     model.fit(X_train, y_train)
#
#     # 加载测试数据
#     X_test = []
#     for filename in os.listdir('test/'):
#         if filename.endswith('.txt'):
#             with open(os.path.join('test/', filename), 'r') as file:
#                 matrix = file.readlines()
#                 vector = np.array([int(pixel) for row in matrix for pixel in row.strip()])
#                 X_test.append(vector)
#
#     X_test = np.array(X_test)
#     X_test = scaler.transform(X_test)  # 使用相同的标准化器
#
#     # 进行预测
#     predictions = model.predict(X_test)
#
#     # 保存结果
#     result_df = pd.DataFrame(predictions, columns=['num'])
#     result_df.to_csv('result.csv', index=False)
#
#     # print("结果已保存到 result.csv")
#
# if __name__ == '__main__':
#     # 设置编码为 UTF-8
#     # sys.stdout.reconfigure(encoding='utf-8')
#     main()

# # V1 优化
# import os
# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# from sklearn.svm import LinearSVC
# from numba import njit
#
# @njit
# def str_to_int_array(s):
#     n = len(s)
#     arr = np.empty(n, dtype=np.int8)
#     for i in range(n):
#         arr[i] = ord(s[i]) - ord('0')
#     return arr
#
# def load_data(data_dir):
#     features = []
#     labels = []
#     for filename in os.listdir(data_dir):
#         if filename.endswith('.txt'):
#             # 提取标签
#             label = int(filename.split('_')[0])
#             # 读取文件内容并转换为特征向量
#             with open(os.path.join(data_dir, filename), 'r') as file:
#                 content = file.read().replace('\n', '')
#                 vector = str_to_int_array(content)
#                 features.append(vector)
#                 labels.append(label)
#     return np.array(features), np.array(labels)
#
# def main():
#     # 加载训练数据
#     X_train, y_train = load_data('train/')
#
#     # 数据标准化
#     scaler = StandardScaler()
#     X_train = scaler.fit_transform(X_train)
#
#     # PCA降维
#     pca = PCA(n_components=0.95)
#     X_train = pca.fit_transform(X_train)
#
#     # 训练 SVM 模型
#     model = LinearSVC(C=1, max_iter=10000)
#     model.fit(X_train, y_train)
#
#     # 加载测试数据
#     X_test = []
#     test_filenames = [filename for filename in os.listdir('test/') if filename.endswith('.txt')]
#     test_filenames.sort(key=lambda x: int(x.split('.')[0]))  # 确保顺序正确
#     for filename in test_filenames:
#         with open(os.path.join('test/', filename), 'r') as file:
#             content = file.read().replace('\n', '')
#             vector = str_to_int_array(content)
#             X_test.append(vector)
#
#     X_test = np.array(X_test)
#     X_test = scaler.transform(X_test)
#     X_test = pca.transform(X_test)
#
#     # 进行预测
#     predictions = model.predict(X_test)
#
#     # 保存结果
#     result_df = pd.DataFrame(predictions, columns=['num'])
#     result_df.to_csv('result.csv', index=False)
#
#     # print("结果已保存到 result.csv")
#
# if __name__ == '__main__':
#     main()


# V2 优化
import os
import numpy as np
import pandas as pd
from sklearn.svm import SVC

# Load training data from directory
def load_train_data(directory):
    features, labels = [], []
    for fname in os.listdir(directory):
        if fname.endswith('.txt'):
            label = int(fname.split('_')[0])
            path = os.path.join(directory, fname)
            with open(path, 'r') as f:
                matrix = [list(map(int, line.strip())) for line in f.readlines()]
                features.append(np.ravel(matrix))
                labels.append(label)
    return np.array(features), np.array(labels)

# Load test data from directory
def load_test_data(directory):
    features, names = [], []
    for fname in sorted(os.listdir(directory), key=lambda x: int(x.split('.')[0])):
        if fname.endswith('.txt'):
            names.append(fname)
            path = os.path.join(directory, fname)
            with open(path, 'r') as f:
                matrix = [list(map(int, line.strip())) for line in f.readlines()]
                features.append(np.ravel(matrix))
    return np.array(features), names

# Directory paths
train_dir = 'train/'
test_dir = 'test/'

# Load data
X_train, y_train = load_train_data(train_dir)
X_test, test_names = load_test_data(test_dir)

# Train SVM model
model = SVC(kernel='linear', C=1.0, random_state=0)
model.fit(X_train, y_train)

# Predict test data
y_pred = model.predict(X_test)

# Save results to CSV
pd.DataFrame({'num': y_pred}).to_csv('result.csv', index=False)
