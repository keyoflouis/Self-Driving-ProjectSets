1. 处理图片，（边缘检测+纯度过滤）

2. 获取相机参数（obj和img点）

3. 矫正相机

4. 计算透视矩阵

5. 应用透视矩阵

6. 找线/确定左右车道线（滑动窗口 - 基于上一帧的滑动窗口 - 基于滑动窗口的卷积增强）

7. 计算曲率

---

# 查找车道线

gene_data 生成模拟数据

lane_historgram.py 统计道路中二值像素点的分布

