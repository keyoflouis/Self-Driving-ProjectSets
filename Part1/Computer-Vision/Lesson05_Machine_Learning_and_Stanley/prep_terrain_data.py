#!/usr/bin/python
import random


def makeTerrainData(n_points=1000):
    ##############################################################################
    ### 生成示例数据集
    random.seed(42)
    grade = [random.random() for ii in range(0, n_points)]
    bumpy = [random.random() for ii in range(0, n_points)]
    error = [random.random() for ii in range(0, n_points)]
    y = [round(grade[ii] * bumpy[ii] + 0.3 + 0.1 * error[ii]) for ii in range(0, n_points)]
    for ii in range(0, len(y)):
        if grade[ii] > 0.8 or bumpy[ii] > 0.8:
            y[ii] = 1.0

    ### 将数据集划分为训练集和测试集
    X = [[gg, ss] for gg, ss in zip(grade, bumpy)]
    split = int(0.75 * n_points)
    X_train = X[0:split]
    X_test  = X[split:]
    y_train = y[0:split]
    y_test  = y[split:]

    grade_sig = [X_train[ii][0] for ii in range(0, len(X_train)) if y_train[ii] == 0]
    bumpy_sig = [X_train[ii][1] for ii in range(0, len(X_train)) if y_train[ii] == 0]
    grade_bkg = [X_train[ii][0] for ii in range(0, len(X_train)) if y_train[ii] == 1]
    bumpy_bkg = [X_train[ii][1] for ii in range(0, len(X_train)) if y_train[ii] == 1]

    # training_data = {"fast":{"grade":grade_sig, "bumpiness":bumpy_sig}
    #         , "slow":{"grade":grade_bkg, "bumpiness":bumpy_bkg}}


    # 快速的坡度值和颠簸值
    grade_sig = [X_test[ii][0] for ii in range(0, len(X_test)) if y_test[ii] == 0]
    bumpy_sig = [X_test[ii][1] for ii in range(0, len(X_test)) if y_test[ii] == 0]
    # 缓慢的坡度和颠簸值
    grade_bkg = [X_test[ii][0] for ii in range(0, len(X_test)) if y_test[ii] == 1]
    bumpy_bkg = [X_test[ii][1] for ii in range(0, len(X_test)) if y_test[ii] == 1]

    test_data = {"fast":{"grade":grade_sig, "bumpiness":bumpy_sig}
            , "slow":{"grade":grade_bkg, "bumpiness":bumpy_bkg}}

    return X_train, y_train, X_test, y_test
    # return training_data, test_data