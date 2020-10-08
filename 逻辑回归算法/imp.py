import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize


# 显示二维图形【两个类别散点图】
def plot_data(X, y):
    pos = np.where(y == 1)  # 找到y==1的坐标位置
    neg = np.where(y == 0)  # 找到y==0的坐标位置
    # 作图
    plt.figure(figsize=(15, 12))
    plt.plot(X[pos, 0], X[pos, 1], 'ro')  # red o
    plt.plot(X[neg, 0], X[neg, 1], 'bo')  # blue o
    plt.title(u"two sandiantu")
    plt.show()


# 添加高阶次的特征向量，将原来的的特征向量[x1,x2]映射为[1,x1,x2,x1^2,x1*x2,x2^2]
# 通过在特征值中增加这些复杂的多项式，可得到更复杂的决策边界，而不仅仅是用直线分开的正负样本
def mapFeature(X1, X2):
    degree = 2  # 映射的最高次方
    out = np.ones((X1.shape[0], 1))  # 映射后的结果数组（取代X）
    '''
    这里以degree=2为例，映射为1,x1,x2,x1^2,x1*x2,x2^2
    '''
    for i in np.arange(1, degree + 1):
        for j in range(i + 1):
            temp = X1 ** (i - j) * (X2 ** j)  # 矩阵直接乘相当于matlab中的点乘.*
            out = np.hstack((out, temp.reshape(-1, 1)))  # -1代表的意思就是，我不知道可以分成多少行，但是我的需要是分成3列，多少行我不关心
    return out


# 代价函数
def costFunction(initial_theta, X, y, inital_lambda):
    m = len(y)
    J = 0
    h = sigmoid(np.dot(X, initial_theta))  # 计算h(z)
    theta1 = initial_theta.copy()  # 因为正则化从theta1开始，不包含theta0，所以复制一份，置theta[0]值为0
    theta1[0] = 0

    temp = np.dot(np.transpose(theta1), theta1)
    J = (-np.dot(np.transpose(y), np.log(h)) - np.dot(np.transpose(1 - y),np.log(1 - h)) + temp * inital_lambda / 2) / m  # 正则化的代价方程
    return J


# 计算梯度
def gradient(initial_theta, X, y, inital_lambda):
    m = len(y)
    grad = np.zeros((initial_theta.shape[0]))

    h = sigmoid(np.dot(X, initial_theta))  # 计算h(z)
    theta1 = initial_theta.copy()
    theta1[0] = 0  # 为啥第一个theta是0

    grad = np.dot(np.transpose(X), h - y) / m + inital_lambda / m * theta1  # 正则化的梯度
    return grad


# S型函数
def sigmoid(z):
    h = np.zeros((len(z), 1))  # 初始化，与z的长度一置
    h = 1.0 / (1.0 + np.exp(-z))
    return h


# 预测
def predict(X, theta):
    m = X.shape[0]  # 行数（样本数m）
    # p = np.zeros((m, 1))  #初始化预测向量（不初始化也没什么毛病）
    p = sigmoid(np.dot(X, theta))  # 计算预测的结果，是个概率值

    # 便利每个样本的预测结果，大于0.5的为1，小于0.5的为0
    for i in range(m):
        if p[i] > 0.5:  # 概率大于0.5预测为1，否则预测为0
            p[i] = 1
        else:
            p[i] = 0
    return p

# 画决策边界
def plotDecisionBoundary(theta, X, y):
    pos = np.where(y == 1)  # 找到y==1的坐标位置
    neg = np.where(y == 0)  # 找到y==0的坐标位置

    print(pos)
    print(neg)

    # 作图
    plt.figure(figsize=(15, 12))
    plt.plot(X[pos, 0], X[pos, 1], 'ro')  # red o
    plt.plot(X[neg, 0], X[neg, 1], 'bo')  # blue o
    plt.title(u"decision boundary")

    # u = np.linspace(30,100,100)
    # v = np.linspace(30,100,100)

    u = np.linspace(-1, 1.5, 50)  # 根据具体的数据，这里需要调整
    v = np.linspace(-1, 1.5, 50)

    z = np.zeros((len(u), len(v)))
    for i in range(len(u)):
        for j in range(len(v)):
            z[i, j] = np.dot(mapFeature(u[i].reshape(1, -1), v[j].reshape(1, -1)), theta)  # 计算对应的值，需要map

    z = np.transpose(z)
    plt.contour(u, v, z, [0, 0.01], linewidth=2.0)  # 画等高线，范围在[0,0.01]，即近似为决策边界
    # plt.legend()
    plt.show()


fileName = "data2.txt"
data = np.loadtxt(fileName, delimiter=',', dtype=np.float64)
# print(data)

# 提取出样本的特征值和结果值
X = data[:, 0:-1]
y = data[:, -1]

# 画出当前csv数据的散点分布图
# plot_data(X, y)

X = mapFeature(X[:, 0], X[:, 1])  # 取第一列，第二列；添加高阶次的特征向量，将原来的的特征向量[x1,x2]映射为[1,x1,x2,x1^2,x1*x2,x2^2]

initial_theta = np.zeros((X.shape[1], 1))  # 初始化theta为1
initial_lambda = 0.0001  # 初始化正则化系数，一般取0.01,0.1,1.....

'''
- 调用scipy中的优化算法fmin_bfgs（拟牛顿法Broyden-Fletcher-Goldfarb-Shanno）来得到最优参数向量theta
- costFunction代价函数是自己实现的一个求代价的函数，
- initial_theta表示初始化参数的值,
- fprime指定costFunction的梯度
- args是其余测参数，以元组的形式传入，最后会将最小化costFunction的theta返回 
'''
result = optimize.fmin_bfgs(costFunction, initial_theta, fprime=gradient, args=(X, y, initial_lambda))

p = predict(X, result)  # 预测

# np.float64(p == y)返回的是向量，可用np.mean计算平均值
print(u'在训练集上的准确度为%f%%' % np.mean(np.float64(p == y) * 100))  # 与真实值比较，p==y返回True，转化为float

_X = data[:, 0:-1]
_y = data[:, -1]
# plotDecisionBoundary(result, _X, _y)  # 画决策边界
