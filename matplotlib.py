import matplotlib.pyplot as plt
import numpy as np

#simple plotting
x = np.logspace(-1,1,20)#orlinspacef or linear
y1 = x**2.0
y2 = x**1.5
plt.loglog(x,y1,"bo-",linewidth=2,markersize=6, label="first")#.plot for linear
plt.loglog(x,y2,"gs-",linewidth=2,markersize=6,label="second")
plt.xlabel("X")
plt.ylabel("Y")
plt.axis([0.5,10.5,-5,105])
plt.legend(loc="upper left")
plt.savefig("myplot.png")


#subplots
x = np.random.gamma(2,3,100000)
plt.figure()
plt.subplot(221)#three arguments row/column/location
plt.hist(x,bins=30,normed=True)
plt.subplot(222)
plt.hist(x,bins=30,normed=True, cumulative=True)
plt.subplot(223)
plt.hist(x,bins=30,normed=True,cumulative=True, histtype="step")
plt.subplot(224)
plt.hist(x,bins=100,normed=True)
