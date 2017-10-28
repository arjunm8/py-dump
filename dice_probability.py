# -*- coding: utf-8 -*-
"""
Created on Sat Jun 24 15:58:17 2017

@author: Arjun
"""

import random
import numpy as mp
import matplotlib.pyplot as plt
import time

start_time = time.clock()
x= np.random.randint(1,7,(1000000,10))
y = np.sum(x,axis=1)
plt.hist(y)

end_time=time.clock()
total_time=end_time-start_time
print(total_time)
