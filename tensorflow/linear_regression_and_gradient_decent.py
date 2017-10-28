# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 11:09:49 2017

@author: Arjun
"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

'''
#basic example
#note a range not arrange
x = np.arange(0,5,0.1)
m=1
b=0
#m is the slope and b is line intercept
y= m*x + b 

plt.plot(x,y) 
plt.ylabel('Dependent Variable')
plt.xlabel('Indepdendent Variable')
'''

'''
When we perform an experiment and gather the data, or if we already have a dataset
and we want to perform a linear regression, what we will do is adjust a simple 
linear model to the dataset, we adjust the "slope" and "intercept" parameters to
 the data the best way possible, because the closer the model comes to describing
 each ocurrence, the better it will be at representing them.
'''


#example for y = 3x + 2
x_data = np.random.rand(100).astype(np.float32)
#the final predictions for a,b should be similar to this(3,2)
y_data = 3 * x_data + 2
#create some gaussian noise look up **vectorize and ***lambda down below
# the lambda function takes y as argument and returns y + some random noise(mean 0, standard dev 0.1)
#vectorize is used to pass y values to the lambda func for each value in y_data
y_data = np.vectorize(lambda y: y + np.random.normal(loc=0.0, scale=0.1))(y_data)
#zip is used to combine arrays
#print(zip(x_data,y_data) [0:5])

#scatter needs separate color and marker arguments
#plt.scatter(x_data,y_data,color="r",marker=".")
#or
#plt.plot(x_data,y_data,"r.")


#First, we initialize the variables a and b, with any random guess, and then we define the linear function:
#experiment with values to see results
a = tf.Variable(1.0)
b = tf.Variable(0.2)
#the linear function
#multiple regression also possible by using more variables(no shit.)
y = a * x_data + b
'''
In a linear regression, we minimize the squared error of the equation that we want to adjust minus the target values (the data that we have), so we define the equation to be minimized as loss.
To find Loss's value, we use __tf.reduce_mean()__. **This function finds the mean of a multidimensional tensor**, and the result can have a diferent dimension(reduced).
'''
loss = tf.reduce_mean(tf.square(y - y_data))

'''
Then, we define the optimizer method. Here we will use a simple gradient descent with a learning rate of 0.5:

Now we will define the training method of our graph, what method will we use for minimizing the loss? We will use the tf.train.GradientDescentOptimizer.
.minimize()__ will minimize the error function of our optimizer, resulting in a better model.

'''
#tldr train's task will be to minimize the loss value, learning rate of optimizer is 0.5.
optimizer = tf.train.GradientDescentOptimizer(0.5)
#use minimize method of optimizer to "loss"
#assuming optimizer adjusts a,b because they're tf variables
#assumption correct and can be tested by turning b into a const
#useful for mutliple regression too, can be tested by adding more variable dimensions
train = optimizer.minimize(loss)


#then we init variables before executing the graph
init = tf.global_variables_initializer()

# Launch the graph.
sess = tf.Session()
sess.run(init)


#time to start optimization and run graph

train_data = []

# Fit the line.
#run 100 training iteration and save every 5th one
for steps in range(200):
    #from index 1 coz train will print Null
    evals = sess.run([train,a,b])[1:]
    if steps%5==0:
        print(steps,evals)
        train_data.append(evals)
       
plt.figure(figsize=(10,10))
cr, cg, cb = (0.0, 1.0, 0.0)
for f in train_data:
    #autistic way for using dynamic colors for each iteration
    #the color is supposed to transition 
    cb += 1.0 / len(train_data)
    cg -= 1.0 / len(train_data)
    if cb > 1.0: cb = 1.0
    if cg < 0.0: cg = 0.0
    
    #unpacking current value of a,b
    [a, b] = f
    #you could do this without loop and  to just get the final adjusted line
    #vec_y = np.vectorize(lambda x: sess.run(a)*x + sess.run(b))(x_data)
    #using lambda for generating y values for each x value using the current iterations a,b
    vec_y = np.vectorize(lambda x: a*x + b)(x_data)
    #uses connected lines if marker,color etc isn't specified
    line = plt.plot(x_data, vec_y)
    #using setproperty to set line color
    plt.setp(line, color=(cr,cg,cb))

plt.plot(x_data, y_data, 'r.')


'''
When more than one independent variable is present the process is called multiple linear regression. When multiple dependent variables are predicted the process is known as multivariate linear regression.

The very known equation of a simple linear model is

Y=aX+b

Where Y is the dependent variable and X is the independent variable, and a and b being the parameters we adjust. a is known as "slope" or "gradient" and b as "intercept". You can interpret this equation as Y being a function of X, or Y being dependent of X.

If you plot the model, you will see it is a line, and by adjusting the "slope" parameter you will change the angle between the line and the independent variable axis, and the "intercept parameter" will affect where it crosses the dependent variable axis.

Simple linear relations were used to try to describe and quantify many observable physical phenomena, the easiest to understand are speed and distance traveled:

Distance Traveled = Speed * Time + Initial Distance

Speed = Acceleration * Time + Initial Speed

They are also used to describe properties of different materials:

Force = Deformation * Stiffness 

Heat Transfered = Temperature Difference * Thermal Conductivity 

Electrical Tension (Voltage) = Electrical Current * Resistance

Mass =  Volume * Density

**vectorize example
Define a vectorized function which takes a nested sequence of objects or numpy arrays
as inputs and returns an single or tuple of numpy array as output. 
>>> def myfunc(a, b):
...     "Return a-b if a>b, otherwise return a+b"
...     if a > b:
...         return a - b
...     else:
...         return a + b

>>> vfunc = np.vectorize(myfunc)
the array is values for A and 2 is B
>>> vfunc([1, 2, 3, 4], 2)
array of returned values
array([3, 4, 1, 2])

***lambda
Python supports the creation of anonymous functions (i.e. functions that are not bound to a name) at runtime, using a construct called "lambda". This is not exactly the same as lambda in functional programming languages, but it is a very powerful concept that's well integrated into Python and is often used in conjunction with typical functional concepts like filter(), map() and reduce().

This piece of code shows the difference between a normal function definition ("f") and a lambda function ("g"):
	>>> def f (x): return x**2
... 
>>> print f(8)
64
>>> 
>>> g = lambda x: x**2
>>> 
>>> print g(8)
64
'''