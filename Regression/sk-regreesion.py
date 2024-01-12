from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
#载入数据
data = np.genfromtxt("data.csv", delimiter=",")

#numpy中的newaxis函数给x_data新增加了一个维度，使其从一维的数组变成了100*1的array
#否则linearregression会报错
x_data = data[:,0,np.newaxis]
y_data = data[:,1,np.newaxis]

model=LinearRegression()
model.fit(x_data,y_data)

plt.plot(x_data, y_data, 'b.')
plt.plot(x_data, model.predict(x_data), 'r')
plt.show()