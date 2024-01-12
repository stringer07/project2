import numpy as np
import matplotlib.pyplot as plt

#输入数据
#异或问题需要输入 x0（偏置项），x1,x2,x1*x2,x1^2,x2^2
X = np.array([[1,0,0,0,0,0],
              [1,1,1,1,1,1],
              [1,1,0,0,1,0],
              [1,0,1,0,0,1]])
#标签
Y = np.array([[-1],
              [-1],
              [1],
              [1]])

#权值初始化，6行1列，取值范围-1到1
W = (np.random.rand(6,1)-0.5)*2
print("Intiallize W: \n{}".format(W))
#学习率设置
lr = 0.11
#神经网络输出
O = 0

def update():
    global X,Y,W,lr
    O = np.dot(X,W)
    W_C = lr*(X.T.dot(Y-O))/int(X.shape[0])
    W = W + W_C

def caculate(x,root):
    a=W[5]
    b=W[2]+W[3]*x
    c=W[4]*(x**2) +W[1]*x+W[0]
    if root==1:
        return((-b-np.sqrt(b**2-4*a*c))/(2*a))
    if root==2:
        return((-b+np.sqrt(b**2-4*a*c))/(2*a))

for i in range(100000):
    update()
    O=np.dot(X,W)
    if ((np.abs(O - Y))<0.01).all():
        print("model convergence")
        break

O=np.dot(X,W)
print("After literations W =\n {}".format(W))
print("Finally output:\n{}".format(O))
print("The y: \n{}".format(Y))

#正样本
x1 = [1,0]
y1 = [0,1]
#负样本
x2 = [0,1]
y2 = [0,1]

xdata=np.linspace(-0.1,1.1)
plt.figure()
plt.plot(xdata,caculate(xdata,1),'r')
plt.plot(xdata,caculate(xdata,2),'r')
plt.scatter(x1,y1,c='b')
plt.scatter(x2,y2,c='y')
plt.show()