---
layout:     post
title:      Linear regression and Logistic regression
subtitle:   The concrete analysis of the algorithm
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

It seems that linear regerssion is the most easy understanding but also powerful model in machine learning. For our Chinese students, we learn the simplify version in our high school, which is just like the Normal Equation.

The model of linear regression is shown:

$$f(x_{i}) = \sum_{i=1}^{m} \omega x + b$$

The cost function is:

$$J(x) = \frac{1}{m}\sum_{i=1}^{m}(f(x_{i}) - y_{i})^2 + \lambda\sum_{i=1}^{m} x$$

The main idea of the linear regression is to find the best parameter $\omega$ to make our cost function to be least. The two main algorithm to calculate the $\omega$ will be shown.

### gradient decent

Since the final goal is to ruduce the cost and let this procedure run as fast as possible, so it is natural to let it go along its gradient. And our algorithm is to update $\omega$ with the gradinet we compute and learning rate $\alpha$ to control its convergence steps.

There are 2 main steps to achieve gradinet decent in the loop iterations

1. construct the cost function J

2. change the $\omega$ to reduce the cost function

$$\omega = \omega - \alpha\frac{dJ}{dx}$$

### normal equation

The following equation is called *Normal Equation*:

$$\omega = (X^{T}X)^{-1}X^{T}y$$

With this equation, $\omega$ can be computed easily. However, we should compute the inverse of the matrix, whose complexity is O($n^{3}$). It will take much more time than *Gradient Decent* if the number of the samples is big. So this method only fit when the data is not very big.
