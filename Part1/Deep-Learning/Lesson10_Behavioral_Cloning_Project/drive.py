import argparse
import base64
import json

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from io import BytesIO

from exceptiongroup import catch
from flask import Flask
import tensorflow as tf

# 使用tensorflow.keras代替直接引用keras
from tensorflow.keras.models import model_from_json
import cv2

sio = socketio.Server()
app = Flask(__name__)
model = None


def resize(img):
    # 保持与训练时相同的尺寸
    return img.resize((320,160), Image.Resampling.LANCZOS)  # 注意 PIL 是 (宽,高)


def cut_top_portion(image):
    return np.array(image)#[15:]


def convert_to_HLS(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)


def normalize(image_data):
    return (image_data / 255.0) - 0.5  # 简化归一化计算


@sio.on('telemetry')
def telemetry(sid, data):
    print("--------- connecting ---------")
    imgString = data["image"]
    image = Image.open(BytesIO(base64.b64decode(imgString)))

    try:
        # 预处理流水线
        processed = resize(image)
        processed = cut_top_portion(processed)
        processed = convert_to_HLS(processed)
        processed = normalize(processed)
        print("Processed image shape:", processed.shape)
    except Exception:
        print("!!!!!! Processe image wrong !!!!!!")

    # 添加批处理维度
    prediction = model.predict(processed[None, ...], verbose=0)
    print("Prediction shape:", prediction.shape)

    steering = float(prediction[0][0])  # 假设模型输出是二维数组
    throttle = 0.2

    print("收到图像数据，尺寸:", image.size)
    print("预测转向角:", steering)

    send_control(steering, throttle)


@sio.on('connect')
def connect(sid, environ):
    print("Client connected:", sid)
    send_control(0, 0)


def send_control(steering, throttle):
    sio.emit("steer", {
        'steering_angle': str(steering),
        'throttle': str(throttle)
    }, skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help='Path to model.json')
    args = parser.parse_args()

    # 模型加载优化
    # 在模型加载部分添加异常处理
    try:
        with open(args.model, 'r', encoding='utf-8') as f:
            model = model_from_json(json.load(f))
    except UnicodeDecodeError:
        print(f"⚠️ 检测到二进制文件，尝试加载完整模型...")
        model = tf.keras.models.load_model(args.model)
    except Exception as e:
        print(f"模型加载失败: {str(e)}")
        exit(1)


    model.compile(optimizer='adam', loss='mse')  # 显式编译模型
    print("\nModel input shape:", model.input_shape)

    # 中间件配置
    app = socketio.Middleware(sio, app)

    # 服务器配置
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)