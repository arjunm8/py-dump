# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 10:16:36 2017

@author: Arjun

"""

'''tf follows a dataflow approach where the graph is created
first then it's executed, it's got a c++ backend hence the speed'''


import tensorflow as tf

'''
# Initialize two constants
x1 = tf.constant([1,2,3,4])
x2 = tf.constant([5,6,7,8])
# Multiply
#use tf.matmul for matrix multiplication
result = tf.multiply(x1, x2)
# Print the result
print(result)
# Intialize the Session
sess = tf.Session()
# Print the result
print(sess.run(result))
# Close the session
sess.close()

# Initialize Session and run `result`
with tf.Session() as sess:
  output = sess.run(result)
  print(output)
  
# adding 2 nums
a = tf.constant([1])
b = tf.constant([2])
c = tf.add(a,b)
session = tf.Session()
with tf.Session() as session:
   result = session.run(c)
   print(result)
'''

'''
other ops
tf.multiply(x, y)
tf.div(x, y)
tf.square(x)
tf.sqrt(x)
tf.pow(x, y)
tf.exp(x)
tf.log(x)
tf.cos(x)
tf.sin(x)
'''

'''
#counter with variables
state = tf.Variable(0)
one = tf.constant(1)
new_value = tf.add(state,one)
#tf.assign used to update variable
update = tf.assign(state,new_value)
#variables need to be init before running
init_op = tf.initialize_all_variables()
with tf.Session() as session:
    session.run(init_op)
    print(session.run(state))
    for _ in range(3):
        session.run(update)
        print(session.run(state))
'''

'''
#placeholder used for feeding data from outside the model
#must define datatype
a = tf.placeholder(tf.float32) 
#or tf.global_variables_initializer  
init = tf.initialize_all_variables()    
#wont run yet coz no value with b
b = a*2
# must feed data in the form of a dict
with tf.Session() as session:
    session.run(init)
    result = session.run(b,feed_dict={a:3.5})
    print(result)
# since data in tensorflow is passed in the form of multi-d arrays
#we can pass any kind of tensor through the placeholders to get the
#answer to the simple multiplication operation
dictionary = {a:[[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]]}
with tf.Session() as sess:
    sess.run(init)
    result = sess.run(b,feed_dict=dictionary)
    print(result)
'''

'''
#fibonacci 1
f = [tf.constant(1),tf.constant(1)]

for i in range(2,10):
    temp = f[i-1] + f[i-2]
    f.append(temp)

with tf.Session() as sess:
    result = sess.run(f)
    print result
    
#fibonacci2

a=tf.Variable(0)
b=tf.Variable(1)
temp=tf.Variable(0)
c=a+b

update1=tf.assign(temp,c)
update2=tf.assign(a,b)
update3=tf.assign(b,temp)

init_op = tf.initialize_all_variables()
with tf.Session() as s:
    s.run(init_op)
    for _ in range(15):
        print(s.run(a))
        s.run(update1)
        s.run(update2)
        s.run(update3)
'''