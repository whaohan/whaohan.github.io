---
layout:     post
title:      Matplotlib
subtitle:   Matplotlib 的常用操作
date:       2020-03-21
author:     翁浩瀚
catalog: true
tags:
    - python 
    - Matplotlib
    - Machine Learning
---
```python
import matplotlib.pyplot as plt
```

#### 基本绘图操作

```python

plt.plot(x,y,'color and shape')
#dot shape . , + o v s p x  

#line shape - -- -. :  


plt.bar(bar_position,bar_height,bar_width)

plt.barh(bar_position,bar_height,bar_width)

plt.hist(data,range=(,),bin=n)

plt.scatter(x,y)

plt.boxplot(data)  

plt.show()

```

#### 图形优化

```python

plt.xticks(rotation=angle)

plt.yticks(rotation=angle)

plt.grid()

plt.xlim(start,end)

plt.ylim(start,end)

plt.xlabel("")

plt.ylabel("")

plt.title("")

plt.legend(loc="parameter")
#parameter:best, upper right, right, lower right, center right  

#          lower center, upper center, center ...  


fig = plt.figure()

subfigi = fig.add_subplot(n,m,i)

```
