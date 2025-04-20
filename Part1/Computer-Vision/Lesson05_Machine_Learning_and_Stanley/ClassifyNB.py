from sklearn.naive_bayes import GaussianNB


def classify(features_train, labels_train):

    ### 导入 sklearn 库中的 GaussianNB 模块
    ### 创建分类器
    ### 在训练特征和标签上拟合分类器
    ### 返回拟合好的分类器

    clf = GaussianNB()
    clf.fit(features_train,labels_train)
    return clf