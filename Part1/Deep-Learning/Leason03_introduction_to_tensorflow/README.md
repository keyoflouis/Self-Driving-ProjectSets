**注释中都是项目给出的Tensorflow代码，均为Tensorflow API版本1。（现已弃用）**

---

### 环境配置

使用官方给出的yml创建环境时，下载了tensorflow后重复下载keras会出现不兼容问题

- Windows上使用conda配置tensorflow环境，
  
  - `pip install tensorflow-gpu==2.9`支持cuda，**Windows上tensorflow版本>2.9的版本无法找到GPU**

- Windows用因为网络问题可能自动下载cudatoolkit失败,
  
  - 手动下载： `cuda install cudatoolkit==11.6.0`

- numpy使用 `conda install numpy==1.26`

- cudnn使用 `conda install cudnn==8.2`