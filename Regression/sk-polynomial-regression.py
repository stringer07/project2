import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

data = np.genfromtxt("job.csv", delimiter=",")
x_data = data[1:,1]
y_data = data[1:,2]
x_data = x_data[:,np.newaxis]
y_data = y_data[:,np.newaxis]

# 定义多项式回归,degree的值可以调节多项式的特征
x_poly =  PolynomialFeatures(degree=3).fit_transform(x_data)# 特征处理
lin_reg = LinearRegression()# 定义回归模型
lin_reg.fit(x_poly, y_data)# 训练模型

'''
plt.plot(x_data, y_data, 'b.')
plt.plot(x_data, lin_reg.predict(poly_reg.fit_transform(x_data)), c='r')#不连续，因为画图只取了10个点（x_data）
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
'''

# 画图
plt.plot(x_data, y_data, 'b.')
x_test = np.linspace(1,10,1000)
x_test = x_test[:,np.newaxis]
plt.plot(x_test, lin_reg.predict( PolynomialFeatures(degree=3).fit_transform(x_test)), c='r')#连续，x取了1000个点
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()