# 切片+运算
import numpy as np

arr_big = np.array([[1, 2, 3, 4, 5, 6], [7, 8, 9, 1, 2, 3]])
arr_small = np.array([[1, 2, 3], [4, 5, 6]])

# 矩阵相乘
print(np.matmul(arr_big[:, 0:3:1], arr_small.T))
# 逐元素相乘
print(arr_big[0, 0:3:1] * arr_small)

# 计算行列式，方阵
arr_det = np.array([[1,2,3],[4,5,6],[7,8,9]])
print(np.linalg.det(arr_det))

# 广播与扩写
arr_row = np.array([1,2,3,4,5,6])
arr_col = np.array([[1],[2],[3],[4],[5],[6]])
print(arr_col*arr_row)