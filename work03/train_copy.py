import os
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.preprocessing import StandardScaler
import sys

def load_data(data_dir):
    features = []
    labels = []
    for filename in os.listdir(data_dir):
        if filename.endswith('.txt'):
            label = int(filename.split('_')[0])
            with open(os.path.join(data_dir, filename), 'r') as file:
                matrix = file.readlines()
                vector = np.array([int(pixel) for row in matrix for pixel in row.strip()])
                features.append(vector)
                labels.append(label)
    return np.array(features), np.array(labels)

def main():
    # 加载训练数据
    X_train, y_train = load_data('train/')

    # 数据标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    # 训练 SVM 模型
    model = svm.SVC(gamma='scale')
    model.fit(X_train, y_train)

    # 加载测试数据
    X_test = []
    for filename in os.listdir('test/'):
        if filename.endswith('.txt'):
            with open(os.path.join('test/', filename), 'r') as file:
                matrix = file.readlines()
                vector = np.array([int(pixel) for row in matrix for pixel in row.strip()])
                X_test.append(vector)

    X_test = np.array(X_test)
    X_test = scaler.transform(X_test)

    # 进行预测
    predictions = model.predict(X_test)

    # 保存结果
    result_df = pd.DataFrame(predictions, columns=['num'])
    result_df.to_csv('result.csv', index=False)

    # 输出结果
    print("结果已保存到 result.csv")

if __name__ == '__main__':
    # 设置编码为 UTF-8
    sys.stdout.reconfigure(encoding='utf-8')
    main()
