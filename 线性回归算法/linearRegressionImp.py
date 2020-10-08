# numpy大型、多维数组上执行数值运算
import numpy as np
from matplotlib import pyplot as plt

fileName = "data.txt"
data = np.loadtxt(fileName, delimiter=',', dtype=np.float64)

# 矩阵的切片操作，从矩阵中分离x和y
X = data[:, 0:-1]
y = data[:, -1]

col = data.shape[1]  # data.shape查看矩阵的维度
X_norm = np.array(X)

mu = np.mean(X_norm, 0)  # 0对各列求均值，1对各行求平均值(结果是一行两列)
sigma = np.std(X_norm, 0)  # 0对各列求标准差，1对各行求标准差(结果是一行两列)

# print(mu)

# print(sigma)

# 归一化
for i in range(X.shape[1]):  # 遍历列，共2列
    X_norm[:, i] = (X_norm[:, i] - mu[i]) / sigma[i]  # （当前值-平均值）/标准差

# 画个图看下归一化feature特征向量后的情况
# plt.scatter(X_norm[:, 0], X_norm[:, 1])
# plt.show()
X_norm = np.hstack((np.ones((X.shape[0], 1)), X_norm))  # 在X前加一列1
#print(X_norm)
# 初始化theta
theta = np.zeros((col, 1))
#print(theta)
y = y.reshape(-1, 1)

m = len(y)
n = len(theta)
temp = np.matrix(np.zeros((n, 400)))  # 暂存每次迭代计算的theta，转化为矩阵形式
J_history = np.zeros((400, 1))  # 记录每次迭代计算的代价值

# 梯度下降
for i in range(400):  # 遍历迭代次数
    h = np.dot(X_norm, theta)  # 计算内积，matrix可以直接乘,求得hypothesis预测值
    temp[:, i] = theta - ((0.01 / m) * (np.dot(np.transpose(X_norm), h - y)))  # 梯度的计算
    theta = temp[:, i]
    # J_history[i]存储的每迭代一次后代价函数的值的变化，利用画图工具可以展示梯度下降的cost fun值的变化
    J_history[i] = (np.transpose(X_norm * theta - y)) * (X_norm * theta - y) / (2 * m)  # 调用计算代价函数

x = np.arange(1, 401)
plt.plot(x, J_history)
plt.xlabel(u"iterator nums")
plt.ylabel(u"cost value")
plt.title(u"the change by iterator nums")
plt.show()
