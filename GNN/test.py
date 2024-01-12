import torch
import networkx as nx 
import matplotlib.pyplot as plt 

#画图的函数
def visualize_graph(G,color):
    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])
    nx.draw_networkx(G,pos=nx.spring_layout(G,seed=42),with_labels=False,node_color=color,cmap="Set2")
    plt.show()
#画点的函数
def visualize_embedding(h,color,epoch=None,loss=None):
    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])
    h=h.detach().cpu().numpy()
    plt.scatter(h[:,0],h[:,1],s=140,c=color,cmap="Set2")
    if epoch is not None and loss is not None:
        plt.xlabel(f'Epoch: {epoch},Loss: {loss.item():.4f}',fontsize=16)
    plt.show

#空手道俱乐部，每个图34个点（只有一个图），每个点特征是一个34维向量 
#对每一个点进行分类任务 
from torch_geometric.datasets import KarateClub
dataset=KarateClub()
print(f'Dataset: {dataset}:')
print('================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of fetures: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')



