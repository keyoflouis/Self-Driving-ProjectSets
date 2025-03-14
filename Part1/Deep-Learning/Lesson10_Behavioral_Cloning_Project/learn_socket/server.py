# 导入事件驱动并发库（用于处理高并发网络请求）
import eventlet
# 关键补丁：替换标准库的阻塞IO操作为非阻塞版本（如socket模块）
# 这使得原生代码（如Flask）能够与eventlet协同工作，实现异步处理
eventlet.monkey_patch()

# 导入Web框架和实时通信库
from flask import Flask
import socketio

# 创建Socket.IO服务器实例，配置异步模式为eventlet
# async_mode指定使用eventlet作为底层异步引擎，以支持高并发实时连接
sio = socketio.Server(async_mode='eventlet')
# 创建Flask应用实例，用于处理常规HTTP请求
app = Flask(__name__)
# 将Flask的WSGI应用包装为Socket.IO应用，使两者共享同一个服务器端口
# 这样同一个端口既能处理HTTP请求，也能处理WebSocket连接
app.wsgi_app = socketio.WSGIApp(sio, app.wsgi_app)

# 注册客户端连接事件处理器
# - sid: 客户端唯一会话ID
# - environ: 包含连接环境信息的字典（如HTTP头、查询参数等）
@sio.event
def connect(sid, environ):
    print(f"客户端连接: {sid}")  # 新客户端连接时触发
    # 此处可添加业务逻辑：用户认证、会话初始化等

# 注册客户端断开事件处理器
@sio.event
def disconnect(sid):
    print(f"客户端断开: {sid}")  # 客户端断开时触发
    # 此处可添加清理逻辑：释放资源、更新状态等

# 注册自定义请求事件处理器（事件名为'request'）
# - data: 客户端发送的消息体（自动解析为字典）
# 必须返回可序列化的数据（如字典），作为响应返回客户端
@sio.event
def request(sid, data):
    print(f"收到来自 {sid} 的请求: {data}")
    # 示例处理逻辑：提取message字段并转为大写
    message = data.get('message', '')  # 安全获取字段，避免KeyError
    processed_data = message.upper()  # 业务处理（此处为示例逻辑）
    # 构造响应格式（遵循RESTful风格）
    return {
        'status': 'success',  # 状态标识
        'result': processed_data,  # 处理结果
        'original': data  # 可选：返回原始数据供客户端校验
    }
@sio.event
def sayHi(sid,data):
    print("\n hello in server \n")

    ret="hello from server"+data.get('message','')

    return {
        'status':'success',
        'result':ret
    }
@sio.on("tele")
def tele(sid,data):

    print('on tele,',data.get('message',''))

    return {
        'status':'success',
        'result':True
    }



# 主程序入口
if __name__ == '__main__':
    # 配置服务器监听参数
    listen_address = ('', 5000)  # 绑定所有网卡，端口5000

    # 创建事件监听器（eventlet封装了socket监听）
    listener = eventlet.listen(listen_address)

    # 启动eventlet WSGI服务器（每个请求在独立greenthread中处理）
    # - 支持高并发：使用协程而非线程，适合IO密集型场景
    # - 自动处理WebSocket升级握手：得益于socketio的中间件
    eventlet.wsgi.server(listener, app)