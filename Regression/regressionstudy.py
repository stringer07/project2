import numpy as np
import matplotlib.pyplot as plt
from numpy import exp

def com_error(b,k,x_data,y_data):
    totalerror=0
    for i in range(0,len(x_data)):
        totalerror+=(y_data[i]-(k*x_data[i]+b))**2
    return totalerror/(2*float(len(x_data)))

def Rsquare(b,k,x_data,y_data):
    SSE=0
    SST=0
    y_tot=0
    for i in range(len(y_data)):
        y_tot+=y_data[i]
    m=float(len(x_data))
    y_avg=y_tot/m
    for i in range(len(x_data)):
        SSE+=(y_data[i]-(k*x_data[i]+b))**2
    for i in range(len(x_data)):
        SST+=(y_data[i]-y_avg)**2
    rsquare=1-(SSE/SST)
    return rsquare
def Rsquareexp(a,b,x_data,y_data):
    SSE=0
    SST=0
    y_tot=0
    for i in range(len(y_data)):
        y_tot+=y_data[i]
    m=float(len(x_data))
    y_avg=y_tot/m
    for i in range(len(x_data)):
        SSE+=(y_data[i]-(a*exp(b*x_data[i])))**2
    for i in range(len(x_data)):
        SST+=(y_data[i]-y_avg)**2
    rsquare=1-(SSE/SST)
    return rsquare

#linear fit
def grad_dc_runner(x_data,y_data,b,k,lr,n):
    m=float(len(x_data))
    for i in range(n):
        b_grad=0
        k_grad=0
        for j in range(0,len(x_data)):
            b_grad+=(1/m)*((k*x_data[j]+b)-y_data[j])
            k_grad+=(1/m)*x_data[j]*((k*x_data[j]+b)-y_data[j])
        b=b-(lr*b_grad)
        k=k-(lr*k_grad)
        if i%10==0:
            print("After {} times iterations b={:.4f},k={:.4f},R_square={:.5f}".format(i,b,k,Rsquare(b,k,x_data,y_data)))
    return b,k

#expfit
def expfit(x_data,y_data,a,b,lr,n):
    m=float(len(x_data))
    for i in range(n):
        a_grad=0
        b_grad=0
        for j in range(len(x_data)):
            a_grad+=(1/m)*(a*exp(b*x_data[i])-y_data[i])*exp(b*x_data[i])
            b_grad+=(1/m)*(a*exp(b*x_data[i])-y_data[i])*exp(b*x_data[i])*a*x_data[i]
        a=a-(lr*a_grad)
        b=b-(lr*b_grad)
        if i%10==0:
            print("After {} times iterations a={:.4f},b={:.4f},R_square={:.5f}".format(i,a,b,Rsquareexp(a,b,x_data,y_data)))
    return a,b

data=np.genfromtxt("data.csv",delimiter=",")
x_data=data[:,0]
y_data=data[:,1]
lr=1e-5
b=1
k=0
a=1
n=100
print("Starting a = {0}, b = {1}, R_square = {2}".format(a, b, Rsquareexp(a, b, x_data, y_data)))
print("Running...")
a,b=expfit(x_data,y_data,b,k,lr,n)
plt.plot(x_data, a*exp(b*x_data),'b')
plt.plot(x_data,y_data,'r.')
plt.show()
