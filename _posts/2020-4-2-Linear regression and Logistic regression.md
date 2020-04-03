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

$$h(x_{i}) = \sum_{i=1}^{m} \omega x + b$$

The cost function is:

$$J(x) = \frac{1}{m}\sum_{i=1}^{m}(f(x_{i}) - y_{i})^2 + \lambda\sum_{i=1}^{m} x$$

The main idea of the linear regression is to find the best parameter $\omega$ to make our cost function to be least. The two main algorithm to calculate the $\omega$ will be shown.

And the second term of the cost function is called regulazatin term, which is used to prevent the overfitting in the machine learning.

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

## Logistic Regression

In my opinion, logistic regression are similar to the linear regression.

While linear regression is expected to output a value fits the model, logistic regression hopes to output a boolean value which determine the event is true or false.

The difference between them is the cost function, which is:

$$J = - \sum_{i=1}^{m}(y\log{h(x_{i})}+(1-y)\log(1-h(x_{i})))$$

It can show the difference between the prediction of the model $h(x)$ and the lable of the given data in a real number. After that, we can use gradient decent to compute the best parameter $\omega$ and $b$.

## At the last

I just use the most simple ways to describe these two models, and don't give the detils in it, such as how to compute the derivative. But I think that the most important thing is to having a intuition about how these algorithms work and you can look up the manual when you implement them.
