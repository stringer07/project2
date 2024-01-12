import numpy as np
import matplotlib.pyplot as plt

def Rsquare(a0,a1,a2,x_data,y_data):
    SSE=0
    SST=0
    y_tot=0
    for i in range(len(y_data)):
        y_tot+=y_data[i]
    for i in range(len(y_data)):
        SSE+=(y_data[i]-(a0+a1*x_data[i,0]+a2*x_data[i,1]))**2
    m=float(len(y_data))
    y_avg=y_tot/m
    for i in range(len(y_data)):
        SST+=(y_data[i]-y_avg)**2
    rsquare=1-(SSE/SST)
    return rsquare

def multi_linear_fit(x_data,y_data,a0,a1,a2,lr,n):
    m=float(len(y_data))
    for i in range(n):
        a1_grad=0
        a2_grad=0
        a0_grad=0
        for j in range(len(y_data)):
            a0_grad+=(1/m)*((a0+a1*x_data[j,0]+a2*x_data[j,1])-y_data[j])
            a1_grad+=(1/m)*((a0+a1*x_data[j,0]+a2*x_data[j,1])-y_data[j])*x_data[j,0]
            a2_grad+=(1/m)*((a0+a1*x_data[j,0]+a2*x_data[j,1])-y_data[j])*x_data[j,1]
        a0-=lr*a0_grad
        a1-=lr*a1_grad
        a2-=lr*a2_grad
        if i%10000==0:
            print("After {} times iterations a0={:.4f},a1={:.4f},a2={:.4f},R_square={:.5f}".format(i,a0,a1,a2,Rsquare(a0,a1,a2,x_data,y_data)))
    return a0,a1,a2

data=np.genfromtxt("Delivery.csv",delimiter=",")
x_data=data[:,:-1]
y_data=data[:,-1]
a0=0
a1=0
a2=0
lr=1e-4
n=100000
a0,a1,a2=multi_linear_fit(x_data,y_data,a0,a1,a2,lr,n)

# 生成网格矩阵
x0=np.linspace(min(x_data[:,0]),max(x_data[:,0]),1000)
x1=np.linspace(min(x_data[:,1]),max(x_data[:,1]),1000)
x0,x1=np.meshgrid(x0,x1)
z = a0 + x0*a1 + x1*a2

# 画3D图
ax = plt.figure().add_subplot(111, projection = '3d') 
ax.scatter(x_data[:,0], x_data[:,1], y_data, c = 'r', marker = 'o', s = 100) #点为红色三角形  
ax.plot_surface(x0, x1, z)

#设置坐标轴  
ax.set_xlabel('Miles')  
ax.set_ylabel('Num of Deliveries')  
ax.set_zlabel('Time')  
  
#显示图像  
plt.show()  