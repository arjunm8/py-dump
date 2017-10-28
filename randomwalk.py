# -*- coding: utf-8 -*-
"""
Created on Sat Jun 24 17:12:00 2017

@author: Arjun
"""
import numpy as np
import matplotlib.pyplot as plt

x0 = np.array([[0],[0]])
delta_x = np.random.normal(0,1,(2,5))

X = np.concatenate((x0,np.cumsum(delta_x,axis=1)),axis=1)

plt.plot(X[0],X[1],"ro-")
