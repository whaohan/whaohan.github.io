---
layout:     post
title:      pytorch教程
subtitle:   The Main Architecture of pytorch
date:       2020-07-06
author:     翁浩瀚
catalog: true
tags:
    - Deep Learning
    - pytorch
---

### 前言

我又来了hhh这次来整理一波pytorch的基本操作，方便查阅。更详细的使用可以访问[pytorch的中文文档](https://pytorch.apachecn.org/docs/1.4/)

### 基本架构

pytorch的结构主要分成三个层次。

第一层是tensor的一些基本操作，有点类似于numpy;

第二层是Autograd的使用，让我们远离无趣的矩阵求导；

第三层是module的认知和使用，其中封装了大量的神经网络的常用模块；

### tensor的基本使用

#### tensor构造

可在函数的参数中，添加`dtype = torch.xxx`确定数据类型(Autograd的要求一般是torch.float)

```python
# 初始化空矩阵

torch.empty(x,y)

# 随机矩阵

torch.rand(x,y)

# 正态分布

torch.randn(x,y)

# 从0到n-1的随机整数排列

torch.randperm(n)

# 0矩阵

torch.zeros(x,y)

# 恒同矩阵

torch.eye(n)

# 均匀采样的一维张量

# steps指个数

torch.linspace(start, end, steps)
# step指距离

torch.arange(start, end, step)

# log刻度均匀的一维张量

torch.logspace(start, end, steps)

# 从已有数据生成,共享内存空间

torch.tensor(array)
torch.from_numpy(ndarray)

# 与tensor1相同形状的随机tensor

torch.randn_like(tensor1, dtype=torch.float)
```

#### tensor操作

```python

# 两种运算方法

# 赋值给第三者

tensor1 + tensor2
# 将2加到1上去

tensor1.add_(tensor2)

# 加法与平均

torch.sum(tensor1, dim=0)
torch.mean(tensor1, dim=0)

# 合并

# 在该维度上直接合并

# (2,2) + (2,2) = (4,2)

torch.cat((tensor1, tensor2), dim=0)
# 新增一个维度并合并

# (2,2) + (2,2) = (2,2,2)

torch.stack((tensor1, tensor2, dim=0))

# 除去维度为1的维度

torch.squeeze()

# 添加多余的一个维度

# 多用于预测时为添加一个假的batch序号

torch.unsqueeze(0)

# 调整维度顺序

# 一次调整多个维度

tensor1.premute(2,0,1)  # (1,2,3) -> (3,1,2)

# 一次互换两个维度

tensor1.transpose(0,1)  # (1,2,3) -> (2,1,3)
```

#### 获取本身属性

```python
# 形状

tensor.size()

# 数值

# 只有一个元素

tensor.item()
# 多个元素

tensor.tolist()
```

### Autograd

实现自动求导也非常的简单啦，我们只需要在tensor初始化的时候给他加一个属性`requires_grad = True`，然后他就会自己记录自己的计算图了。

#### 对于一个loss/scalar

我们只需要用一个`loss.backward()`，他就会自己噌噌噌地求导，然后计算过程中的任何需要求导的节点$v$，他的gradient都会被存储在`v.grad`，直接调用即可。

#### 对于一个vector/matrix

这个就比较麻烦一点啦。

我们先想象一下，对于某个计算图，他backprop的时候，我们的导数计算总是等于 local_grad $\times$ upstream-grad , 当我们的终止节点是一个非scalar时，我们就没有办法获得他的 upstream_grad ， 这个时候我们就要手动输入这个grad啦。也就意味着，我们的代码变成了`loss.backward(upstram_grad)`,对于 grad 的获取就还是一样啦。

#### 关于取消计算图记录

1. 使用这行代码直接取消他自动求导的性质，这样子自然不会记录他的计算图啦
   `tensor.requires_grad_(False)`

2. 使用一个非记录轨迹的块：
   `with torch.no_grad():`

### 神经网络Module的构建

#### 神经网络架构代码示例

啥都别说，我们先贴一个官网的示例代码(LeNet5)，向老前辈致敬！

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 输入图像channel：1；输出channel：6；5x5卷积核

        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 2x2 Max pooling

        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # 如果是方阵,则可以只使用一个数字进行定义

        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # 除去批处理维度的其他所有维度

        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()
print(net)

# 创建优化器(optimizer）

optimizer = optim.SGD(net.parameters(), lr=0.01)
# 在训练的迭代中：

optimizer.zero_grad()   # 清零梯度缓存/初始化

output = net(input)
loss = nn.MSELoss()(y_pred, y_train)
loss.backward()
optimizer.step()    # 更新参数
```

-----------------------------------------------

```python
output:
    Net(
        (conv1): Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1))
        (conv2): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
        (fc1): Linear(in_features=400, out_features=120, bias=True)
        (fc2): Linear(in_features=120, out_features=84, bias=True)
        (fc3): Linear(in_features=84, out_features=10, bias=True)
    )
```

#### 正向传播与反向传播

在定义好 net 之后，我们仅仅需要使用：
    y_pred = net(input)
就可以获得正向传播的输出了。与此同时，反向传播会由Autograd帮我们完成，我们啥都不需要做。

#### 损失函数

几种常用的loss：

```python
torch.nn.L1Loss(size_average=None, reduce=None, reduction='mean')

torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')

torch.nn.CrossEntropyLoss(weight=None, size_average=None,
                ignore_index=-100, reduce=None, reduction='mean')
```

#### 优化器

几种常用的optimizer：

```python
torch.optim.SGD(params, lr=<required parameter>, momentum=0,
                    dampening=0, weight_decay=0, nesterov=False)

torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999),
                    eps=1e-08, weight_decay=0, amsgrad=False)
```
