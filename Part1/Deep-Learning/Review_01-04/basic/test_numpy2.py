import numpy as np

arr_row = np.array([1, 2, 3, 4, 5, 6])

arr_col = np.array([[1], [2], [3], [4], [5]])

# 基于广播机制，可以做到逐元素相乘,
# 所有维度要么相等，要么为1，就可以扩写
# print(arr_row * arr_col)


arr_2row = np.array([[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]])
arr_4row = np.array([[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]])

# 调整 arr_2row 的形状为 (2, 1, 6),提升为3维（奇怪的操作）
arr_2row = arr_2row[:, np.newaxis, :]

# print(arr_2row)

# 现在可以进行广播操作
result = arr_4row * arr_2row
# print(result)

# print(arr_2row[0,0:6:1]*arr_4row)


arr_2row = np.array([[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]])

arr_5row = np.array([[2, 3, 4, 3, 6, 7], [1, 1, 1, 4, 5, 6]])
# print(arr_2row*arr_4row)

bool_index = arr_2row <= arr_5row

print(arr_2row[bool_index])
