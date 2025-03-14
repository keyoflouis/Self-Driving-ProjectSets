import socketio

# 创建Socket.IO客户端
sio = socketio.Client()

try:
    # 连接服务器
    sio.connect('http://localhost:5000')
    print("已连接到服务器")

    # 发送请求并获取响应
    response = sio.call('request', {'message': 'hello from client'})
    print("服务器响应:", response)

    response2 = sio.call('tele',data={'message':"sssssss"})
    print(response2)

    # 断开连接
    sio.disconnect()
    print("已断开连接")
except Exception as e:
    print(f"发生错误: {e}")
finally:
    if sio.connected:
        sio.disconnect()