import numpy as np
import matplotlib.pyplot as plt
from torch import nn,optim
from torch.autograd import Variable
import torch

#生成随机的线性数据
x_data=np.random.rand(100)
noise=np.random.normal(0,0.01,x_data.shape)
y_data=x_data*0.1 +0.2+ noise
#数据可视化
plt.scatter(x_data,y_data)
plt.show()
# 将一维数据变为二维数据，-1为自动匹配值，在此为100，1为一列,也可以用newaxis
#x_data=x_data.reshape(-1,1)
#y_data=y_data.reshape(-1,1)
x_data=x_data[:,np.newaxis]
y_data=y_data[:,np.newaxis]
x_data=torch.FloatTensor(x_data)
y_data=torch.FloatTensor(y_data)


inputs=Variable(x_data)#x_data作为变量，数据的输入
target=Variable(y_data)#y_data作为变量，数据的输出


#构建神经网络模型
#一般把网络中具有可学习参数的层放在__init__()中
class LinearRegression (nn.Module):
    #定义网络结构
    def __init__(self):
        super(LinearRegression,self).__init__()#继承的父类初始化nn.Module，固定动作
        self.fc=nn.Linear(1,1)#输入一个值，输出一个值

    #定义网络计算
    def forward (self,x):
        out=self.fc(x)
        return out
    
#定义模型
model=LinearRegression()
#定义代价函数
los_function=nn.MSELoss()
#定义优化器,利用随机梯度下降法来进行优化运算，学习率为0.1
optimizer=optim.SGD(model.parameters(),lr=0.1)
'''#查看参数
for name,parameters in model.named_parameters():
    print('name:{},parameters:{}'.format(name,parameters))'''

for i in range (1001):
    out=model(inputs)
    #计算loss
    loss=los_function(out,target)
    #梯度清零
    optimizer.zero_grad()
    #计算梯度
    loss.backward()
    #修改权值
    optimizer.step()
    if i%200==0:
        print(i,loss.item())

y_pred=model(inputs)
plt.scatter(x_data,y_data)
plt.plot(x_data,y_pred.data.numpy(),'r-',lw=1)
plt.show()