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
#linearly spaced 100 points from -pi to +pi
x = np.linspace(-np.pi, np.pi, number_of_samples)
#when y=mx+b falls into pure autism
y = 0.5*x+np.sin(x)+np.random.random(x.shape)

plt.figure(figsize=(10,10))
'''
plt.scatter(x,y,marker = '.',color='black') #Plot y-vs-x in dots
plt.xlabel('x-input feature')
plt.ylabel('y-target values')
plt.title('Fig 1: Data for linear regression')
'''
#jumble(100)
random_indices = np.random.permutation(number_of_samples)
#Training set first 70%
x_train = x[random_indices[:70]]
y_train = y[random_indices[:70]]
#Validation set next 15%
x_val = x[random_indices[70:85]]
y_val = y[random_indices[70:85]]
#Test set next 15%
x_test = x[random_indices[85:]]
y_test = y[random_indices[85:]]

#Create a least squared error linear regression object
lin_model = linear_model.LinearRegression()

#sklearn takes the inputs as matrices. Hence we reshape the arrays into column matrices
#here an array like [1,2,3,4] ie shape(1,) has been transformed to shape(100,1) so [[1],[2]....]
x_train_for_line_fitting = np.matrix(x_train.reshape(len(x_train),1))
y_train_for_line_fitting = np.matrix(y_train.reshape(len(y_train),1))

#Fit the line to the training data
lin_model.fit(x_train_for_line_fitting, y_train_for_line_fitting)

#Plotting
plt.scatter(x_train, y_train, color='black')
#reshaping x to (x_length,1) matrix as well since sklearn takes and returns matrices not 1d arrays, 
#(1,x_length) should work too as long as it's the same during fitting
#x is the x-coords, and lin_model.predict returns the y coords based on the x coords and the training data
plt.plot(x.reshape((len(x),1)),lin_model.predict(x.reshape((len(x),1))),color='blue')
plt.xlabel('x-input feature')
plt.ylabel('y-target values')
plt.title('Fig 2: Line fit to training data')

#evaluate mean squared error / MSE
mean_val_error = np.mean( (y_val - lin_model.predict(x_val.reshape(len(x_val),1)))**2 )
mean_test_error = np.mean( (y_test - lin_model.predict(x_test.reshape(len(x_test),1)))**2 )

print ('Validation MSE: ', mean_val_error, '\nTest MSE: ', mean_test_error)
