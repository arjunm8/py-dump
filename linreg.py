# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 09:52:22 2017

@author: Arjun
"""

import numpy as np
from sklearn import linear_model, datasets, tree
import matplotlib.pyplot as plt

number_of_samples = 100
#create some autistic synth data
x = np.linspace(-np.pi, np.pi, number_of_samples)
y = 0.5*x+np.sin(x)+np.random.random(x.shape)
plt.figure(figsize=(10,10))

plt.scatter(x,y,marker = '.',color='black') #Plot y-vs-x in dots
plt.xlabel('x-input feature')
plt.ylabel('y-target values')
plt.title('Fig 1: Data for linear regression')

#jumble(100)
random_indices = np.random.permutation(number_of_samples)
#Training set
x_train = x[random_indices[:70]]
y_train = y[random_indices[:70]]
#Validation set
x_val = x[random_indices[70:85]]
y_val = y[random_indices[70:85]]
#Test set
x_test = x[random_indices[85:]]
y_test = y[random_indices[85:]]

#Create a least squared error linear regression object
lin_model = linear_model.LinearRegression()

#sklearn takes the inputs as matrices. Hence we reshpae the arrays into column matrices
x_train_for_line_fitting = np.matrix(x_train.reshape(len(x_train),1))
y_train_for_line_fitting = np.matrix(y_train.reshape(len(y_train),1))

#Fit the line to the training data
lin_model.fit(x_train_for_line_fitting, y_train_for_line_fitting)

#Plot the line
#plt.scatter(x_train, y_train, color='black')
#reshape to prevent (n,1) dimension issue in numpy
plt.plot(x.reshape((len(x),1)),lin_model.predict(x.reshape((len(x),1))),color='blue')
plt.xlabel('x-input feature')
plt.ylabel('y-target values')
plt.title('Fig 2: Line fit to training data')
plt.show()