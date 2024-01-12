import numpy as np
import matplotlib.pyplot as plt
from torch import nn,optim
from torch.autograd import Variable
import torch

#生成随机的线性数据
x_data=np.linspace(-2,2,200)[:,np.newaxis]
noise=np.random.normal(0,0.2,x_data.shape)
y_data=np.square(x_data)+noise
#数据可视化
plt.scatter(x_data,y_data)
plt.show()
#Array变成torch中的tensor
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
        #1-10-1
        self.fc1=nn.Linear(1,10)#加一个隐藏层, 具有10个神经元
        self.tanh=nn.Tanh()
        self.fc2=nn.Linear(10,1)

    #定义网络计算
    def forward (self,x):
        x=self.fc1(x)
        x=self.tanh(x)
        x=self.fc2(x)
        return x
 
#定义模型
model=LinearRegression()
#定义代价函数
los_function=nn.MSELoss()
#定义优化器,利用随机梯度下降法来进行优化运算，学习率为0.3
optimizer=optim.SGD(model.parameters(),lr=0.3)
'''#查看参数
for name,parameters in model.named_parameters():
    print('name:{},parameters:{}'.format(name,parameters))'''

for i in range (4001):
    out=model(inputs)
    #计算loss
    loss=los_function(out,target)
    #梯度清零
    optimizer.zero_grad()
    #计算梯度
    loss.backward()
    #修改权值
    optimizer.step()
    if i%400==0:
        print(i,loss.item())

y_pred=model(inputs)
plt.scatter(x_data,y_data)
plt.plot(x_data,y_pred.data.numpy(),'r-',lw=3)
plt.show()