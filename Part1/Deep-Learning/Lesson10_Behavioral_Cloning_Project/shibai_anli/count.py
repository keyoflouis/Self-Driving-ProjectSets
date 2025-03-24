import tensorflow as tf
import numpy as np


# 禁用所有数据增强和预处理
def parse_raw_data(line):
    """仅解析中心摄像头原始数据"""
    parts = tf.strings.split(line, ',')
    steering = tf.strings.to_number(parts[3], out_type=tf.float32)
    return steering.numpy()


# 加载原始CSV数据
raw_lines = tf.data.TextLineDataset("data/driving_log.csv").skip(1)  # 跳过标题行
raw_steerings = [parse_raw_data(line) for line in raw_lines.as_numpy_iterator()]

# 计算直方图（保持与之前相同的区间设置）
hist_counts, bin_edges = np.histogram(
    raw_steerings,
    bins=50,
    range=(-1.0, 1.0)  # 与增强数据相同的范围
)

# 打印原始数据分布
print("原始数据（未增强）转向角分布：")
print(f"{'转向区间':<25} | {'样本数量':<8} | {'占比 (%)':<6}")
print("-" * 45)

total_samples = len(raw_steerings)
for i in range(len(hist_counts)):
    left = bin_edges[i]
    right = bin_edges[i + 1]
    count = hist_counts[i]
    percent = (count / total_samples) * 100

    # 突出显示关键区间
    range_str = f"[{left:.4f}, {right:.4f})"
    if abs(left) < 0.1:
        range_str += " (直行)"
    elif left > 0.3:
        range_str += " (急右转)"
    elif right < -0.3:
        range_str += " (急左转)"

    print(f"{range_str:<25} | {count:<8} | {percent:.2f}%")

# 关键统计量
print("\n关键统计：")
print(f"总样本数: {total_samples}")
print(f"中位数: {np.median(raw_steerings):.4f}")
print(f"均值: {np.mean(raw_steerings):.4f}")
print(f"标准差: {np.std(raw_steerings):.4f}")
print(f"|θ| < 0.1 样本占比: {np.mean(np.abs(raw_steerings) < 0.15) * 100:.2f}%")
print(f"|θ| > 0.4 样本占比: {np.mean(np.abs(raw_steerings) > 0.4) * 100:.2f}%")