
# 图嵌入+可视化基本框架

## 1准备工作安装

+ install torch
+ install torch_geometric
+ install sklearn
+ install tensorboardX

## 2导入必要的包

+ node2vec: 
    - from torch_geometric.nn import Node2Vec
+ TSNE:
    - from sklearn.manifold import TSNE
+ SummaryWriter
   - from tensorboardX import SummaryWriter


## 3数据读取
+ 本demo用的数据集是cora，它采集了七种类别的机器学习论文，总共2708篇，1433个关键字。
+ data的数据格式是：
  edge_index=[2, 10556] x=[2708, 1433], y=[2708]


## 4模型构建
```python
model = Node2Vec(data.num_nodes,embedding_dim=128, walk_length=20，context_size=10, walks_per_node=10)
```

## 5模型训练

## 6可视化

6.1 pytorch中自带T-SNE可视化工具

``` python
out = model(torch.arange(data.num_nodes, device=device))
out = TSNE(n_components=2).fit_transform(out.cpu().numpy())#将模型的嵌入向量转换到2维度的tsne嵌入空间中，做一进步的降为，最后可视化

#下面用基本的画图工具matplotlib.pyplot画图即可

y = data.y.cpu().numpy()
plt.figure(figsize=(8, 8))
colors = [
    '#ffc0cb', '#bada55', '#008080', '#420420', '#7fe5f0', '#065535', '#ffd700'
]
for i in range(dataset.num_classes):
    plt.scatter(out[y == i, 0], z[y == i, 1], s=20, color=colors[i])
plt.axis('off')
plt.show()
```

![pytorch_T-SNE](./image/pytorch_tsne.png)
<center>图1 pytorch中TSNE函数可视化图</center>


6.2使用tensorboardX进行可视化

基本框架是：
```python
writer = SummaryWriter(log_dir='./last/exp1', comment='tsne')#log_dir指定文件保存路径，commnet指定文件名称
writer.add_embedding(tag='node2vec',mat=output, metadata=label,global_step=epoch)
writer,close()
'''
tag:数据的名称，不同名称的数据将分别展示
mat:一个矩阵，每行代表特征空间的一个数据点
metadata:一个一维列表，mat中每行数据的label，大小应该和mat行数相同
global_step:训练的step
'''
```
运行结束后，在终端输入tensorboard --logdir=文件路径
打开http://localhost:6006即可看见可视化的图。

在页面中，可视化方式既可以选择PCA也可以选择T-SNE
![tensorboard_TSNE_2D](./image/tensorboard_tsne_2d.png)
<center>图2 tensorboardX中TSNE函数可视化2维图片</center>

![tensorboard_TSNE_2D](./image/tensorboard_tsne_3d.png)
<center>图3 tensorboardX中TSNE函数可视化3维图片</center>







     
  
  