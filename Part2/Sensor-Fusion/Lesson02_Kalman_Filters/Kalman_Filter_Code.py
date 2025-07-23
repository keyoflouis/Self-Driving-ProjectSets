def update(mean1, var1, mean2, var2):
    new_mean = float(var2 * mean1 + var1 * mean2) / (var1 + var2)
    new_var = 1./(1./var1 + 1./var2)
    return [new_mean, new_var]

def predict(mean1, var1, mean2, var2):
    new_mean = mean1 + mean2
    new_var = var1 + var2
    return [new_mean, new_var]

measurements = [5., 6., 7., 9., 10.]  #测量值
motion = [1., 1., 2., 1., 1.] #运动值
measurement_sig = 4. # 测量值的不确定性（方差）
motion_sig = 2. # 运动值的不确定性（方差）
mu = 0. #当前位置
sig = 10000. #当前位置的方差

# 应用卡尔曼滤波器
for n in range(len(measurements)):
    [mu,sig] = update(mu,sig,measurements[n],measurement_sig)
    print( 'input:',[mu,sig])
    [mu,sig] = predict(mu,sig,motion[n],motion_sig)
    print('predict:',[mu,sig])

print(mu, sig)

