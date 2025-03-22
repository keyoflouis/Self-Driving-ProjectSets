### 环境

drive.py 

socketio库编写的服务器与udacity模拟器的socketio版本不同时容易出现不兼容。

- 我的drive.py实现中使用到了`tensorflow 2.x` 的API

- [socketio环境参考](https://zhuanlan.zhihu.com/p/356440288)  

---

- [x] 需要熟悉数据集管道的构建

# 最终结果：use_mut_train_o3.py

---

tarin.py 自定义神经网络

resize_train.py 自定义神经网络+裁剪数据

use_mutcamera_train.py 使用了多个摄像头,

use_mut_train_o1.py 使用了管道预处理数据集,NVIDIA的神经网络

我注意到drive.py对图片进行过预处理，但我的神经网络内部已经处理过图片，修改后重新尝试

use_mut_train_o2.py 使用特征融合,采用多次训练，在最后一个弯道表现不佳。

use_mut_train_o2_5.py deepseek添加了注意力机制,

use_mut_train_o3.py 成功在0.18的油门下行驶一圈

use_mut_train_o3_5.py （失败）

use_mut_train_o4.py 尝试对数据分布进行处理（失败）

----

train2.py 去掉了速度特征，尝试更均衡的数据分布：直行 30 % ，大转弯9%。

模型训练过程损失下降顺利，但实际运行失败

train3.py 尝试调整模型配置，使用余弦退火，运行失败

----

2025.3.22

行为克隆项目卡着了

问题1，训练数据目前已经基本解决，训练数据仍可以进一步加强，采用各种光照。

问题2，模型采用NVIDIA的架构合理么（大概是的）。那么我对模型不够熟悉，无法更改模型结构

（当前关键问题）问题3，模型编译参数，优化器配置等，我不熟悉（当前关键问题），试训练效率低下

问题4，模型的单次预测我无法知晓具体内部内容

----

2025.3.23

解决问题2，问题3，问题4。

若是仍然无法做到，就不继续纠结其任何他的问题，去看别人的答案
