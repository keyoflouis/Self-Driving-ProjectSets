from test import *
import time


def load_pathset(path, target_dirs):
    ''' 返回车辆图片和非车辆图片的路径集合 '''

    images = []
    for dir_name in target_dirs:
        start_path = os.path.join(path, dir_name)
        for root, _, files in os.walk(start_path):
            for file in files:
                if file.lower().endswith(('png')):
                    images.append(os.path.join(root, file))

    cars = []
    notcars = []
    for image in images:
        if 'non-vehicles' in image:
            notcars.append(image)
        else:
            cars.append(image)

    return cars, notcars


def load_data(vehicle_list, non_vehicle_list):
    car_features = [extract_features(image) for image in vehicle_list]
    non_car_features = [extract_features(image) for image in non_vehicle_list]

    # 堆叠向量
    X = np.vstack((car_features, non_car_features)).astype(np.float64)
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(non_car_features))))

    # 分割数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

    # 归一化
    X_scaler = StandardScaler().fit(X_train)

    # 应用标准器
    X_train = X_scaler.transform(X_train)
    X_test = X_scaler.transform(X_test)

    with open('feature_label.p', 'wb') as f:
        pickle.dump([X_train, y_train, X_test, y_test, X_scaler], f)


def train_svc():
    t = time.time()
    # 使用线性 SVC
    svc = LinearSVC()

    t = time.time()
    # 训练 SVC
    svc.fit(X_train, y_train)
    print("训练 SVC 耗时 {:.2f}s".format(time.time() - t))

    # 保存训练好的 SVM
    joblib.dump(svc, "train_svc.m")



if not os.path.exists('feature_label.p'):
    path = "../IGNORE/"
    target_dirs = {'non-vehicles', 'vehicles'}
    cars, notcars = load_pathset(path,target_dirs)
    load_data(cars,notcars)

with open('feature_label.p', 'rb') as f:
    X_train, y_train, X_test, y_test, X_scaler = pickle.load(f)

# 打印相关信息
print('训练样本数量：', len(X_train))
print('测试样本数量：', len(X_test))
print('特征向量长度：', len(X_train[0]))

if not os.path.exists("train_svc.m"):
    train_svc()
svc = joblib.load("train_svc.m")

# 测试训练好的 SVC
accuracy = svc.score(X_test, y_test)
print('SVC 的测试准确率 = ', accuracy)