#!/usr/bin/python

""" 补充ClassifyNB.py中的代码，使用sklearn的朴素贝叶斯分类器对地形数据进行分类。

本练习的目标是重现课程视频中展示的决策边界，
并生成一个直观展示决策边界的图表。 """

from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture, output_image
from ClassifyNB import classify
from sklearn.metrics import  accuracy_score

import numpy as np
import pylab as pl

features_train, labels_train, features_test, labels_test = makeTerrainData()

### 训练数据（features_train, labels_train）中混合了"快速"和"慢速"点
### 将它们分开以便在散点图中用不同颜色标识，
### 并直观区分这些点
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii] == 0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii] == 0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii] == 1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii] == 1]

# 你需要补充从ClassifyNB脚本导入的classify函数。
# 确保切换到该代码标签以完成本测验。
clf = classify(features_train, labels_train)

pred_test = clf.predict(features_test)
acc = accuracy_score(labels_test,pred_test)
print(f'acc:{acc}')
### 绘制决策边界并叠加文本点
prettyPicture(clf, features_test, labels_test)
output_image("test.png", "png", open("test.png", "rb").read())
