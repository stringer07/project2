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
        self.fc1=nn.Linear(784,10)#784个输入(28*28个像素)，10个输出
        self.softmax=nn.Softmax(dim=1)#(64,10)对第1个维度（10）进行概率转换

    def forward(self,x):
        #([64,1,28,28])->(64,784)
        x=x.view(x.size()[0],-1)#数据形状变换
        x=self.fc1(x)
        x=self.softmax(x)
        return x

#定义模型
model=Net()
#定义代价函数
mse_loss=nn.CrossEntropyLoss( )
#定义优化器
LR=0.5
optimizer=optim.SGD(model.parameters(),LR)

def train():
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
    print("Test acc: {}".format(correct.item()/len(test_dataset)))

for epoch in range(10):
    print("epoch :",epoch+1)
    train()
    test()