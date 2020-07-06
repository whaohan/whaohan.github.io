---
layout:     post
title:      pytorch教程
subtitle:   The Main Architecture of pytorch
date:       2020-05-20
author:     翁浩瀚
catalog: true
tags:
    - Deep Learning
    - pytorch
---

### 前言

我又来了hhh这次来整理一波pytorch的基本操作，方便查阅。更详细的使用可以访问[pytorch的中文文档](https://pytorch.apachecn.org/docs/1.4/)

### 基本架构

pytorch主要分成三个层次。

第一层是tensor，地位上基本等同于能够使用GPU的numpy；

第二层是Variable，具有自动求导等多种方便的功能；

第三层是module，封装了大量的神经网络的常用模块；

```python
import torch
import numpy as np
```

```python
x = 3
y = 4
array = [x,y]
ndarray = np.array(array)
tensor1 = torch.tensor(array)
n = start = end = step = steps =  4
```

### tensor

#### tensor构造，可在参数中附加类似`dtype = torch.float`确定数据类型

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

torch.linspace(start, end, steps) # steps指个数

torch.arange(start, end, step)    # step指距离

# log刻度均匀的一维张量

torch.logspace(start, end, steps)

# 从已有数据,共享内存空间

torch.tensor(array)

torch.from_numpy(ndarray)

# 与tensor1相同形状的随机tensor

torch.randn_like(tensor1, dtype=torch.float)
```

#### tensor操作

```python

# 两种运算方法

tensor1 + tensor2       # 赋值给第三者

tensor1.add_(tensor2)   # 将2加到1上去

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
