from math import *


# 高斯函数
def f(mu, sigma2, x):
    return 1 / sqrt(2. * pi * sigma2) * exp(-.5 * (x - mu) ** 2 / sigma2)

# 最大化即使得x等于mu
print(f(10., 4., 10.))


# 高斯乘法
def update(mean1, var1, mean2, var2):
    new_mean = (var2 * mean1 + var1 * mean2) / (var1 + var2)
    new_var = 1/(1/var1 +1/var2)
    return [new_mean,new_var]

print(update(10.,4.,12.,4.))