import sys
from class_vis import prettyPicture
from prep_terrain_data import makeTerrainData

import matplotlib.pyplot as plt
import copy
import numpy as np
import pylab as pl


# 获取地形数据：训练集特征、训练集标签、测试集特征、测试集标签
features_train, labels_train, features_test, labels_test = makeTerrainData()


########################## SVM #################################
### 这里我们帮你处理了导入语句和SVC创建
from sklearn.svm import SVC
# 创建SVM分类器，使用RBF核函数，正则化参数C=1
clf = SVC(kernel="rbf",C=1)


#### 现在你的工作是训练分类器
#### 使用训练集特征/标签进行拟合
#### 并在测试数据上进行预测

# 用训练数据拟合SVM模型
clf.fit(features_train,labels_train)

# 对测试数据进行预测
pred = clf.predict(features_test)

# 绘制美观的分类结果图
prettyPicture(clf,features_test,pred)



#### 将你的预测结果存储在名为pred的列表中

# 导入准确率评估指标
from sklearn.metrics import accuracy_score
# 计算预测准确率
acc = accuracy_score(pred, labels_test)

# 打印准确率
print(acc)

# 提交准确率的函数
def submitAccuracy():
    return acc