# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 16:01:33 2017

@author: Arjun
"""
# Probability of Class or Theta(y) or p = exp(y)/(1+exp(y)) or 1/(1+e**-x)
#where y is the regression result
#p-values produced between 0 (as y approaches minus infinity) and 1 (as y approaches plus infinity).
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
 
#we load and separate the dataset into xs and ys, and then into training xs and ys and testing xs and ys, (pseudo-)randomly.
iris = load_iris()
iris_x, iris_y = iris.data[:-1,:], iris.target[:-1]
iris_y= pd.get_dummies(iris_y).values
trainX, testX, trainY, testY = train_test_split(iris_x, iris_y, test_size=0.33, random_state=42)









'''
In order to estimate a classification, we need some sort of guidance on what would be the most probable class for that data point. For this, we use Logistic Regression.
Recall linear regression:

Linear regression finds a function that relates a continuous dependent variable, y, to some predictors (independent variables x1, x2, etc.). Simple linear regression assumes a function of the form:

y=w0+w1∗x1+w2∗x2+...

and finds the values of w0, w1, w2, etc. The term w0 is the "intercept" or "constant term" (it's shown as b in the formula below):

Y=WX+b

Logistic Regression is a variation of Linear Regression, useful when the observed dependent variable, y, is categorical. It produces a formula that predicts the probability of the class label as a function of the independent variables.

Despite the name logistic regression, it is actually a probabilistic classification model. Logistic regression fits a special s-shaped curve by taking the linear regression and transforming the numeric estimate into a probability with the following function:

exp(y) = e(e=natural logarithm ^ y)
theta(y) = probability of class,called the logistic function or logistic curve
ProbabilityOfaClass=θ(y)=ey1+ey=exp(y)/(1+exp(y))=p

which produces p-values between 0 (as y approaches minus infinity) and 1 (as y approaches plus infinity). This now becomes a special kind of non-linear regression.

In this equation, y is the regression result (the sum of the variables weighted by the coefficients), exp is the exponential function and θ(y)

is the logistic function, also called logistic curve. It is a common "S" shape (sigmoid curve), and was first developed for modelling population growth.

You might also have seen this function before, in another configuration:

ProbabilityOfaClass=θ(y)=11+e−x

So, briefly, Logistic Regression passes the input through the logistic/sigmoid but then treats the result as a probability
'''