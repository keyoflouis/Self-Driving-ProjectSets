import numpy as np

#### numpy 数组是tensorflow的基础
#arr1 =np.array([1,2,1,3,5])
#arr2 = np.array([2,8,7,4,5])
#
## 这里的乘法是逐元素乘法，并非向量点积
#print(arr2 * arr1)
#
## 向量点积
#print(sum(arr2 * arr1))
#print(sum(a*b for a,b in zip(arr1,arr2)))
#
## numpy 数组变形
#arr3 = np.array([[1,2,3],[4,5,6]])
#reshape_arr3 = arr3.reshape(3,2)
#print(reshape_arr3)
#
#arr4 =np.array([2,1,6])
## 扩写行向量arr4为矩阵，与矩阵arr3逐元素相乘
#print('arr4 * arr3\n',arr4 * arr3,'\n')
#print(np.multiply(arr4,arr3))
#
## 扩写行向量arr4为矩阵，与矩阵arr3逐元素相加
#print('arr4 + arr3\n',arr4 + arr3,'\n')
#print(np.add(arr4,arr3))
#
## 矩阵乘法
#print(np.matmul(arr3,reshape_arr3))




#切片操作，普通Python数组（仅支持一维）,start:end索引+1:step
nor_list = [1,1,2.1,5.7,9,6,1]
print(nor_list[0:len(nor_list)-1:2])


arr5 = np.array([[1,2,3,4,5],[4,5,6,7,8]])
#numpy数组，多维索引
print(arr5[1,2])
#多维切片 start:end索引+1:step
print(arr5[:,0:5:2])





