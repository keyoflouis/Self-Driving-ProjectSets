# import argparse
# import base64
# import json
#
# import numpy as np
# import socketio
# import eventlet
# import eventlet.wsgi
# from PIL import Image
# from io import BytesIO
#
# from exceptiongroup import catch
# from flask import Flask
# import tensorflow as tf
#
# # 使用tensorflow.keras代替直接引用keras
# from tensorflow.keras.models import model_from_json
# import cv2
#
# sio = socketio.Server()
# app = Flask(__name__)
# model = None
#
#
# def resize(img):
#     # 保持与训练时相同的尺寸
#     return img.resize((320,160), Image.Resampling.LANCZOS)  # 注意 PIL 是 (宽,高)
#
#
# def cut_top_portion(image):
#     return np.array(image)#[15:]
#
#
# def convert_to_HLS(img):
#     return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
#
#
# def normalize(image_data):
#     return (image_data / 255.0) - 0.5  # 简化归一化计算
#
#
# @sio.on('telemetry')
# def telemetry(sid, data):
#     print("--------- connecting ---------")
#     imgString = data["image"]
#     image = Image.open(BytesIO(base64.b64decode(imgString)))
#
#     speed = float(data.get("speed", 0.0)) * 0.44704  # 假设原始单位是mph转m/s
#     print(f"当前速度: {speed:.1f}m/s ({speed*2.23694:.1f}mph)")
#
#
#     # try:
#     #     # 预处理流水线
#     #     processed = resize(image)
#     #     processed = cut_top_portion(processed)
#     #     processed = convert_to_HLS(processed)
#     #     processed = normalize(processed)
#     #     print("Processed image shape:", processed.shape)
#     # except Exception:
#     #     print("!!!!!! Processe image wrong !!!!!!")
#
#     # 添加批处理维度
#     prediction = model.predict(image[None, ...], verbose=0)
#     print("Prediction shape:", prediction.shape)
#
#     steering = float(prediction[0][0])  # 假设模型输出是二维数组
#     throttle = 0.2
#
#     print("收到图像数据，尺寸:", image.size)
#     print("预测转向角:", steering)
#
#     send_control(steering, throttle)
#
#
# @sio.on('connect')
# def connect(sid, environ):
#     print("Client connected:", sid)
#     send_control(0, 0)
#
#
# def send_control(steering, throttle):
#     sio.emit("steer", {
#         'steering_angle': str(steering),
#         'throttle': str(throttle)
#     }, skip_sid=True)
#
#
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('model', help='Path to model.json')
#     args = parser.parse_args()
#
#     # 模型加载优化
#     # 在模型加载部分添加异常处理
#     try:
#         with open(args.model, 'r', encoding='utf-8') as f:
#             model = model_from_json(json.load(f))
#     except UnicodeDecodeError:
#         print(f"⚠️ 检测到二进制文件，尝试加载完整模型...")
#         model = tf.keras.models.load_model(args.model)
#     except Exception as e:
#         print(f"模型加载失败: {str(e)}")
#         exit(1)
#
#
#     model.compile(optimizer='adam', loss='mse')  # 显式编译模型
#     print("\nModel input shape:", model.input_shape)
#
#     # 中间件配置
#     app = socketio.Middleware(sio, app)
#
#     # 服务器配置
#     eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
#
#

import argparse
import base64
import json
import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from io import BytesIO
from flask import Flask
import tensorflow as tf

sio = socketio.Server()
app = Flask(__name__)
model = None

# 必须与训练时使用的原始输入尺寸一致
EXPECTED_IMAGE_SHAPE = (160, 320, 3)  # (height, width, channels)


def prepare_inputs(raw_image, speed):
    """将原始数据转换为模型输入格式"""
    # 转换为numpy数组并验证尺寸
    img_array = np.array(raw_image)
    if img_array.shape != EXPECTED_IMAGE_SHAPE:
        raise ValueError(f"输入图像尺寸错误: 应为 {EXPECTED_IMAGE_SHAPE}，实际为 {img_array.shape}")

    # 应用与训练相同的归一化
    normalized = (img_array.astype(np.float32) / 255.0) - 0.5

    # 构建模型输入 (注意顺序与模型定义一致)
    return [
        np.expand_dims(normalized, axis=0),  # 添加batch维度
        #np.array([[speed]], dtype=np.float32)  # 保持二维形状
    ]


@sio.on('telemetry')
def telemetry(sid, data):
    try:
        # 解码原始图像数据
        img_str = data["image"]
        raw_image = Image.open(BytesIO(base64.b64decode(img_str)))

        # 获取原始速度（假设训练时未转换单位）
        speed = float(data.get("speed", 0.0))

        # 准备模型输入（不进行任何预处理）
        model_input = prepare_inputs(raw_image,speed)

        # 执行预测
        steering = float(model.predict(model_input, verbose=0)[0][0])

        # 动态油门控制（示例逻辑）
        base_throttle = 0.2
        #throttle = base_throttle * (1 - abs(steering))  # 转向越大油门越小
        throttle = base_throttle
        send_control(steering, throttle)

        # 打印调试信息
        print(f"Input shape: {np.array(raw_image).shape} | "
              f"Steering: {steering:.3f} | "
              f"Throttle: {throttle:.2f}")

    except Exception as e:
        print(f"处理异常: {str(e)}")
        send_control(0, 0)


@sio.on('connect')
def connect(sid, environ):
    print("客户端连接:", sid)
    send_control(0, 0)


def send_control(steering, throttle):
    sio.emit("steer", {
        'steering_angle': str(steering),
        'throttle': str(throttle)
    }, skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help='包含预处理层的模型路径 (.h5)')
    args = parser.parse_args()

    # 加载模型并验证结构
    try:
        model = tf.keras.models.load_model(args.model)
        print("✔ 成功加载包含预处理层的模型")

        # 验证关键预处理层存在
        required_layers = ['cropping2d', 'resizing']
        model_layers = [layer.name.lower() for layer in model.layers]
        for layer_name in required_layers:
            if layer_name not in model_layers:
                raise ValueError(f"模型缺少必要层: {layer_name}")

    except Exception as e:
        print(f"❌ 模型加载失败: {str(e)}")
        exit(1)

    # 打印输入输出信息
    print("\n模型输入结构:")
    for i, inp in enumerate(model.inputs):
        print(f"Input {i}: {inp.name}, Shape: {inp.shape}")

    print("\n模型输出结构:")
    for o, out in enumerate(model.outputs):
        print(f"Output {o}: {out.name}, Shape: {out.shape}")

    # 启动服务器
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)