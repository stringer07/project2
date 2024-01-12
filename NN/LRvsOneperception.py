import matplotlib.pyplot as plt
import numpy as np

data = np.genfromtxt("LR-testSet.csv", delimiter=",")
X= np.concatenate((np.ones((100,1)),data[:,:-1]),axis=1)
Y= data[:,-1,np.newaxis]
x=data[:,:-1]
y=data[:,-1]

def update():
    global X,Y,W,lr
    O = np.sign(np.dot(X,W)) # shape:(3,1)
    W_C = lr*(X.T.dot(Y-O))/int(X.shape[0])
    W = W + W_C

def plot():
    x0=[]
    y0=[]
    x1=[]
    y1=[]
    for i in range(len(y)):
        if y[i]==1:
            x1.append(x[i,0])
            y1.append(x[i,1])
        else:
            x0.append(x[i,0])
            y0.append(x[i,1])
    scatter0=plt.scatter(x0,y0,c='b',marker="o")
    scatter1=plt.scatter(x1,y1,c='r',marker="x")
    plt.legend(handles=[scatter0,scatter1],labels=['label0','label1'],loc='best')
plot()
plt.show()

#权值初始化，3行1列，取值范围-1到1
W = (np.random.random([3,1])-0.5)*2
print("初始权值为：\n {}".format(W))
#学习率设置
lr = 0.001
#神经网络输出
O = 0

for i in range(10000):
    update()#更新权值
    #print("迭代{}次后的权值为:\n {}".format(i+1,W))#打印当前权值
    O = np.sign(np.dot(X,W))#计算当前输出  
    if(O == Y).all(): #如果实际输出等于期望输出，模型收敛，循环结束
        print('Finished')
        print('epoch:',i+1)
        break

#计算分界线的斜率以及截距
k = -W[1]/W[2]
d = -W[0]/W[2]
print('k=',k)
print('d=',d)

xdata=(-4,4)
plot()
plt.plot(xdata,xdata*k+d,'r')
plt.show()