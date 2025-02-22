import numpy as np


def cross_entropy(Y, P):
    np.float_(Y)
    np.float_(P)

    return -np.sum(Y * np.log(P) + (1 - Y) * np.log(1 - P))


def test_cross_entropy():
    # 测试用例 1
    Y1 = np.array([1, 0, 1, 0])
    P1 = np.array([0.9, 0.1, 0.8, 0.2])
    expected1 = -np.sum(Y1 * np.log(P1) + (1 - Y1) * np.log(1 - P1))
    assert np.isclose(cross_entropy(Y1, P1), expected1), f"Test case 1 failed: {cross_entropy(Y1, P1)} != {expected1}"

    # 测试用例 2
    Y2 = np.array([1, 1, 0, 0])
    P2 = np.array([0.7, 0.6, 0.4, 0.3])
    expected2 = -np.sum(Y2 * np.log(P2) + (1 - Y2) * np.log(1 - P2))
    assert np.isclose(cross_entropy(Y2, P2), expected2), f"Test case 2 failed: {cross_entropy(Y2, P2)} != {expected2}"

    print("所有测试用例通过！")


# 运行测试函数
test_cross_entropy()
