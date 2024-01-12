import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt  

data=genfromtxt("data.csv",delimiter=",")
x_data = data[:,0,np.newaxis]
y_data = data[:,1,np.newaxis]

'''print(np.mat(x_data).shape)#变成了100*1的矩阵
print(np.mat(y_data).shape)'''
# 给样本添加偏置项
#将两个array合并，一个是由ones生成的100*1的全1array（偏置项），另一个是x_data
X_data = np.concatenate((np.ones((100,1)),x_data),axis=1)

# 标准方程法求解回归参数
def weights(xArr, yArr):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr)
    xTx = xMat.T*xMat # 矩阵乘法
    # 计算矩阵的值,如果值为0，说明该矩阵没有逆矩阵
    if np.linalg.det(xTx) == 0.0:
        print("This matrix cannot do inverse")
        return
    # xTx.I为xTx的逆矩阵
    w = xTx.I*xMat.T*yMat
    return w

w = weights(X_data,y_data)
print(w)
# 画图
x_test = np.array([[20],[80]])
y_test = w[0] + x_test*w[1]
plt.plot(x_data, y_data, 'b.')
plt.plot(x_test, y_test, 'r')
plt.show()