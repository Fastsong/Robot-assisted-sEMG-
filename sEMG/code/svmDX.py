import scipy.io as scio
import glob
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# 对于每个人的数据
for i in range(1,10):
    # 加载数据
    
    data0 = scio.loadmat(f'D:\SEMG\data\prodata\ML2\{i}\DX0.mat')
    data1 = scio.loadmat(f'D:\SEMG\data\prodata\ML2\{i}\DX1.mat')
    data2 = scio.loadmat(f'D:\SEMG\data\prodata\ML2\{i}\DX2.mat')

    X = np.vstack((data0['data'],data1['data'],data2['data'])) # 特征数据
    y = np.vstack((data0['label'],data1['label'],data2['label'])) # 标签数据

    y = y.ravel()
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # 定义分类器
    classifiers = {
        'SVM': SVC(),
        'Random Forest': RandomForestClassifier(),
        'K-NN': KNeighborsClassifier(),
        'MLP': MLPClassifier()
    }

    # 训练和评估每个分类器
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(f'Direction {i}, Classifier: {name}')
        print(classification_report(y_test, y_pred))