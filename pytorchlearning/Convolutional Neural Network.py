import torch
from torch import nn,optim
from torchvision import datasets,transforms
from torch.utils.data import DataLoader

#训练集
train_dataset=datasets.MNIST(root='./',train=True,
                             transform=transforms.ToTensor(),#把数据转换成tensor
                             download=True)
#测试集
test_dataset=datasets.MNIST(root='./',train=False,
                             transform=transforms.ToTensor(),#把数据转换成tensor
                             download=True)
# 批次大小（每次训练传入的数据量）
batchsize=64

#装载训练集
train_loader=DataLoader(dataset=train_dataset,
                        batch_size=batchsize,
                        shuffle=True)#打乱数据
#装载测试集
test_loader=DataLoader(dataset=test_dataset,
                        batch_size=batchsize,
                        shuffle=True)

'''
for data in train_loader:
    inputs,lables=data #train_loader中每一个data结构形如[inputs_tensor,lables_tensor]的tensor组成
    print(inputs.shape)#torch.Size([64,1,28,28])
    print(lables.shape)
    break
'''

#定义网络结构
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        #输入，输出通道数，卷积窗口大小，步长，prdding，图像外部填两圈0
        self.conv1=nn.Sequential(nn.Conv2d(1,32,5,1,2),nn.ReLU(),nn.MaxPool2d(2,2))#池化，2*2步长为2的窗口 
        self.conv2=nn.Sequential(nn.Conv2d(32,64,5,1,2),nn.ReLU(),nn.MaxPool2d(2,2))   
        self.fc1=nn.Sequential(nn.Linear(64*7*7,1000),nn.Dropout(p=0.5),nn.ReLU())
        self.fc2=nn.Sequential(nn.Linear(1000,10),nn.Softmax(dim=1))

    def forward(self,x):
        #([64,1,28,28]) 卷积只能对四维数据做运算
        x=self.conv1(x)
        x=self.conv2(x)
        x=x.view(x.size()[0],-1)#全连接层只能对两位数据进行处理
        x=self.fc1(x)
        x=self.fc2(x)
        return x 

#定义模型
model=Net()
#定义代价函数
mse_loss=nn.CrossEntropyLoss( )
#定义优化器
LR=0.001
optimizer=optim.Adam(model.parameters(),LR)

def train():
    #训练状态，dropout起作用
    model.train()
    for i,data in enumerate(train_loader):
        #获得一个批次的数据和标签
        inputs,labels=data
        #获得模型预测效果（64，10）
        out=model(inputs)
        #交叉熵代价函数out(batch,C),lables(batch)
        #对于交叉熵代价函数，数据shape(out,lables)不需要一致，所以不需要独热编码变换
        loss=mse_loss(out,labels)
        #梯度清零
        optimizer.zero_grad()
        #计算梯度
        loss.backward()
        #修改权值
        optimizer.step()

def test():
    #测试状态，dropout不起作用
    model.eval()
    correct=0
    for i,data in enumerate(test_loader):
        #获得一个批次的数据和标签
        inputs,labels=data
        #获得模型预测效果（64，10）
        out=model(inputs)
        #获得最大值，以及最大值所在的位置
        _,predicted=torch.max(out,1)
        #预测正确的数量
        correct+=(predicted==labels).sum()
    print("Test acc: {:.4f}".format(correct.item()/len(test_dataset)))

    correct=0
    for i,data in enumerate(train_loader):
        #获得一个批次的数据和标签
        inputs,labels=data
        #获得模型预测效果（64，10）
        out=model(inputs)
        #获得最大值，以及最大值所在的位置
        _,predicted=torch.max(out,1)
        #预测正确的数量
        correct+=(predicted==labels).sum()
    print("Train acc: {:.4f}".format(correct.item()/len(train_dataset)))

for epoch in range(20):
    print("epoch :",epoch+1)
    train()
    test()
