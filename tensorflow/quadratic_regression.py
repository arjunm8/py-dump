# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 15:22:01 2017

@author: Arjun
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

x_data = np.random.rand(100).astype(np.float32)

#***see why you did this
x_data = x_data[x_data.argsort()]

#y = ax^2 + bx + c
y_data = 5 + 2*x_data + 2*(x_data**2)

y_data = np.vectorize(lambda y: y+np.random.normal(loc=0,scale=0.1))(y_data)

b0 = tf.Variable(3.0)
b1 = tf.Variable(5.0)
b2 = tf.Variable(4.0)
y = b0 + b1*x_data + b2*(x_data**2)

loss = tf.reduce_mean(tf.square(y-y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

train_data = []

for steps in range(200):
    vals = sess.run([train,b0,b1,b2])[1:]
    train_data.append(vals)
plt.figure(figsize=(10,10))
###replacable with bottomblock for hiding training data
for f in train_data:
    [b0,b1,b2] = f
    vec  = np.vectorize(lambda x: b0 + b1*x + b2*(x**2))(x_data)
'''
#  plt.plot(x_data,vec,color='lightblue')
###remove the following 2 lines incase of showing these training steps
[m,b] = vals
vec  = np.vectorize(lambda x: m*x+b)(x_data)'''

''' 
***
apparently the points in the arrays were not in any order and lead to the points being 
connected at random(try removing the argsorts), resulting in pure autism, so you did array[sorted indexes for ascending order]
and that way the points were plotted in ascending order and hence connected properly
'''
print('b0:',b0,' b1:',b1,' b2:',b2)
plt.plot(x_data,vec,color='blue')
plt.plot(x_data,y_data,"r.")
