---
layout:     post
title:      Linear regression and Logistic regression
subtitle:   简单介绍以及代码实现
date:       2020-04-02
author:     翁浩瀚
catalog: true
tags:
    - python 
    - linear regression
    - logistic regression
    - Machine Learning
---
## Linear regression

  线性回归算是机器学习中最简单的模型了吧，在我们高中的时候也学过一元的线性回归，使用的是最小二乘法（也就是我们等下会提到的Normal Equation的简化版本）。
  线性回归的核心的思想就是使用线性函数或者多项式函数来拟合所获得数据变量，从而使我们的输出和目标输出的差距最小。
  通常而言，这个模型有两种算法，分别是gradient decent和normal equation。下面给出算法以及简单的python代码的实现

### gradient decent

