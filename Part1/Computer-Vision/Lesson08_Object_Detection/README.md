### 目标检测

- draw_bboxes.py 画出蓝色的框

- templ_match.py 模板匹配

- color_histogram.py 直方图RGB通道的颜色统计

- explore_color_spaces.py 探索不同颜色空间的照片下的表现，（待实现封装）

- spatial_bin.py 输入RGB图像，HLS，输出转换为HLS的展平图像

- car_notcar.py 统计数据集

- get_hog.py 从图片中计算得到hog特征，histogram of gradient车辆的部分pattern

- norm_shuffle.py 拼接hog特征和hsv颜色特征，标准化.

- car_features.py 使用空间颜色信息和颜色通道的分布直方图训练一个汽车分类器


----

20讲述的sci-kit中hog

计算图片每个单元格的梯度直方分布图，使用block进行归一化，
最后拼接得到的一维数组即是hog特征向量

##### hog内的归一化：

- 在HOG中，一个block内的归一化过程如下：

- 首先计算block内所有cell的直方图连接起来形成一个长向量v

- 计算这个向量的L2范数：L2 = √(v₁² + v₂² + ... + vₙ²)

- 对向量进行归一化：v_normalized = v / (L2 + ε) (ε是一个很小的常数，防止除以0)

##### 伽马归一化

在计算梯度前对图像所有像素开根号

##### 这篇文章提到的`hog`函数的主要参数包括：

1. **`orientations`**：  
   - 指定每个单元格内梯度方向直方图的分箱数量，通常为6到12。

2. **`pixels_per_cell`**：  
   - 指定计算梯度直方图的单元格大小，通常为正方形（如8x8像素）。

3. **`cells_per_block`**：  
   - 指定局部区域（块）的大小，用于归一化直方图计数，通常为2x2单元格。

4. **`transform_sqrt`**：  
   - 可选参数，用于伽马归一化，可以减少光照变化的影响，但图像不能包含负值。

5. **`visualize`**：  
   - 布尔值，如果为`True`，则返回两个值，(hog_features, hog_image)，
   否则只返回hog_features

6. **`feature_vector`**：  
   - 布尔值，如果为`True`，则将HOG特征展平为一维向量。

这些参数可以帮助你灵活地提取HOG特征，并根据需求调整特征提取的细节。

        
