**注释中都是项目给出的Tensorflow代码，均为Tensorflow API版本1。（现已弃用）**

---

# 构建一个基本的字母分类器

### 重要实验

- `Linarfunction`的 `test.py`

  - 一个最基本的训练模型

- `Mini-batch` 的 `useMinibatch-inMNist` 的 `quiz.py`

  - 使用SDG进行随机小批量梯度下降



---

### 环境配置

使用官方给出的yml创建环境时，下载了tensorflow后重复下载keras会出现不兼容问题

- Windows上使用conda配置tensorflow环境，
  
  - `pip install tensorflow-gpu==2.9`支持cuda，**Windows上 tensorflow>2.9 版本无法找到GPU**

- Windows用因为网络问题可能自动下载cudatoolkit失败,
  
  - 手动下载： `cuda install cudatoolkit==11.6.0`

- numpy使用 `conda install numpy==1.26`

- cudnn使用 `conda install cudnn==8.2`

---
### 总结

- 配环境的坑太多

- 教程中一直都是以挖空的形式来教学，最好在 `Leason07`之前借助`tensorflow`搓一个神经网络。

- `Miniflow` 是一个合格的轮子