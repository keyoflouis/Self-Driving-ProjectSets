#!/usr/bin/python

# from udacityplots import *
import warnings

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use('agg')

import matplotlib.pyplot as plt
import pylab as pl
import numpy as np


# import numpy as np
# import matplotlib.pyplot as plt
# plt.ioff()

def prettyPicture(clf, X_test, y_test):
    x_min = 0.0;
    x_max = 1.0
    y_min = 0.0;
    y_max = 1.0

    # 绘制决策边界。为此，我们将为网格 [x_min, m_max]x[y_min, y_max] 中的每个点分配一种颜色。
    h = .01  # 网格中的步长
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # 将结果放入颜色图中
    Z = Z.reshape(xx.shape)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    plt.pcolormesh(xx, yy, Z, cmap=pl.cm.seismic)

    # 同时绘制测试点
    grade_sig = [X_test[ii][0] for ii in range(0, len(X_test)) if y_test[ii] == 0]
    bumpy_sig = [X_test[ii][1] for ii in range(0, len(X_test)) if y_test[ii] == 0]
    grade_bkg = [X_test[ii][0] for ii in range(0, len(X_test)) if y_test[ii] == 1]
    bumpy_bkg = [X_test[ii][1] for ii in range(0, len(X_test)) if y_test[ii] == 1]

    plt.scatter(grade_sig, bumpy_sig, color="b", label="fast")
    plt.scatter(grade_bkg, bumpy_bkg, color="r", label="slow")
    plt.legend()
    plt.xlabel("bumpiness")
    plt.ylabel("grade")

    plt.savefig("test.png")


import base64
import json
import subprocess


def output_image(name, format, bytes):
    image_start = "BEGIN_IMAGE_f9825uweof8jw9fj4r8"
    image_end = "END_IMAGE_0238jfw08fjsiufhw8frs"
    data = {}
    data['name'] = name
    data['format'] = format
    data['bytes'] = base64.encodebytes(bytes).decode('ascii')
    print (image_start + json.dumps(data) + image_end)