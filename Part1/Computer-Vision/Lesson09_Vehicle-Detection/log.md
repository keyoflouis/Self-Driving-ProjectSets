
1

image_test.py 使用不同的缩放大小对图片重采样并合成，效果一般，性能也差。

尝试用不同的窗口尺寸识别车辆，但是训练器尺寸不一导致报错。 需要重新训练一个分类器？

2

标准器-》特征尺寸-》读入图片尺寸，读入图片的尺寸是64*64，需要降采样训练一个32大小的模型。

在video_test 中先分散使用查看效果。

封装函数？单纯封装么，还是可以自由调节窗口大小。能够自由调节窗口大小么。

3

不能自由调节窗口大小，并且每次窗口大小变化后都需要新训练一个模型？不合理

窗口大小作为参数调整后会导致特征输入尺寸不对，该怎样使得任意窗口大小的输入尺寸一样呢。

find_car_heatmap中实时计算了不同尺寸大小的xy方向的窗口数量，也许在这段展开前进行缩放操作会有效？

`hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()`

但降分辨率与升分辨率的算法该如何实现呢？

应该这样做么，文中的scale有什么用呢

4 

scale 是用来调整检测窗口的大小的，因此不必去编写降分辨率和升分辨率算法。

5

使用不同的scale可以检测不同大小的车辆，选择在特定位置范围内检测，并设立合适的阈值

测试画框，并调整不同大小的窗口的识别区域，设立合适的阈值。

6

打印了不同窗口的识别区域，窗口大小，确定将窗口分设为2.75,2.5,1.5,1，但效果不佳。这合适么？

如果不合适该设为多少，如果合适是否还有别的步骤需要实现？

首先即为应该设为多少，

（大问题为应该如何使用不同的窗口来识别车辆，目前是应该将每个窗口设置为多少大小）

答案里的实现是使用不同的scale值实现的不同尺寸的窗口么。

看答案或者看教程。

先看教程，思考还有什么地方有漏再做打算。

7

窗口搜索是为了直接找到这个车辆的位置，而不是为了生成热图最后筛选出车辆？

调整了识别之后依然有问题，依然会识别到道路左侧，但我之前设置的尺寸似乎不会轻易识别到左侧？

看看昨天的实现？

8

下午看答案吧，首先观察别人的实验，然后尝试复刻，然后使用YOLO尝试，

找到一个还不错的答案，正在查看README文档。deepseek翻译太蠢了老是擅自修改，输出的格式也经常不是markdown能显示的格式

项目是jvpyter形式的，需要翻译为中文.尝试用chrome的deepseek翻译，只能翻译一部分。还是复制粘贴吧，，当前得到了合适的翻译。

9

在草稿中尝试运行他的代码,其scikit的joblib已经独立为一个库，moviepy也更新，label因该无太大变化。

逻辑应该有所改变，先尝试运行代码.展示图片这里逻辑没有问题，能够运行

训练似乎也没有问题,进入滑动窗口的实现了。

现在看起来他的窗口实现也没有什么太特别的，也许是我的分类器的问题当初有思考过，但没有去打印实际的分类器的准确率

视频处理报错，似乎是VideoFileClip里的fl_image被移出导致的

尝试用我的视频处理保存的视频似乎有格式问题

moviepy的类当中没有fl和fl_image这是怎么回事，包损坏么？

moviepy的中文官网不出意料的烂，谷歌找到了，2.7之后movepy不支持editor，直接全部导入即可使用iterator可以获取到每一帧

10 

视频效果挺一般的，但比我的好很多。明天照着调一下参数，然后就尝试用YOLO来实现。

比较疲惫，先从训练开始吧.

他先把数据集提取,保存到了一个pickle文件中然后训练svc，

11

当我想到融合这个答案的方法和我自己的方法的时候，心里有点抵触感？

检测图片的流程-》

imread，归一化，生成滑动窗口（这里采用390,430？放弃检测近距离车辆？），

搜索窗口（没有整体求hog，而是完全逐窗口操作？）

他是如何实现不同窗口大小的呢，如果放弃整体求hog，就可以在sliding_window中直接生成不同尺寸的窗口。

如果使用整体求hog，则只能通过控制缩放，并且多计算几次寻找车辆的函数，才能做到。

12

他的滑动窗口实现和官方的不一样。

x，y start_stop设置与官方一样，都计算了xy方向的跨度。

然后官网的通过覆盖率和窗口的xy方向像素大小计算了x/y方向的步长，他这里的nx_buffer是什么呢

nx_buffer代表在滑动窗口过程中，窗口在水平方向重叠像素数。

这里的nx_pix_per_steps和官网的代表一样的意思，都是x方向每一步的像素数量。

nx_windows 是每种窗口在x方向下的窗口数量.

ny_pix_per_step以及nx_pix_per_step都是列表，因为要遍历多种窗口，但这里的nx都是靠公式计算的。

这个ny是什么意思呢，为什么要这么计算呢，y方向只是简单的切条，也就是没有覆盖的划分。

（这种差异主要是因为代码只在 x 方向实现了“滑动窗口+重叠”的精细控制，
而 y 方向只是简单地把区域均分成和窗口数量一样的条带，每条带用一种窗口大小。）

现在应该是所有应该看懂的都看懂了，整体来说他的效率高，且代码相对简单。

13

选择特征来实现车辆检测就此为止吧。后续准备YOLO8的实现

