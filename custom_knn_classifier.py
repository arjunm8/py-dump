# -*- coding: utf-8 -*-
"""
Created on Sat Jul  1 16:18:17 2017

@author: Arjun
"""
import numpy as np
import random
import scipy.stats as ss
import matplotlib.pyplot as plt

def majority_vote(votes):
    """will create a dict of list item counts
    and return the one with max count"""
    vote_counts = {}
    #if you can't even understand this part then don't even try to read ahead
    for vote in votes:
        if vote in vote_counts:
            vote_counts[vote] +=1            
        else:
            vote_counts[vote] =1
    #save max count value
    max_count = max(vote_counts.values())
    winners = []
    for vote, count in vote_counts.items():
        #create a list of items with with max count
        if count  == max_count:
            winners.append(vote)
    #return a random list item
    return random.choice(winners)
 
    
def majority_vote_short(votes):
    """same shit as majority_vote but doesn't use random"""
    mode,count = ss.mstats.mode(votes)
    return mode


def distance(p1,p2):
    """returns distance between (x,y) and (x',y')
    uses formula: sqrt( (x-x')^2 + (y-y')^2 )"""
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)


def find_nearest(p,points, k=5):
    """loops over the user defined point p with all the values in points[]
    and returns the k number of closes item indexes"""
    #initialize empty distances array with zeroes
    #set it the same size as of the number of rows(hence shape[0])
    distances = np.zeros(points.shape[0])
    #loop over distances and points[] 
    #save distance for each p<->points[:] in distances
    for i in range(len(distances)):
        distances[i] =distance(p,points[i])
    #sort distances in ascending order and return the indexes in an array
    index = np.argsort(distances)
    #return k number of closest distance indexes
    return index[0:k]


def knn_predict(p, points, outcomes,k=5):
    """saves the k number of indexes from nearest neighbors and return the
    majority class/ most frequent element type of points[] as defined in
    outcomes(which should be the same length as points[] to represent
    classes/types for each point). hence determining the class of point p"""
    ind = find_nearest(p,points, k)
    return majority_vote(outcomes[ind])


def generate_synth_data(n=50):
    """generate n x 2 sets of of synthetic/random points with classes 1,0"""
    #join two points[] arrays along axis0(rows) each of normal distributions of
    # mean 0,deviation=1 and mean 1,deviation 1 and shape(n,2)
	#Random variates rvs(loc=0, scale=1, size=1, random_state=None)
    points = np.concatenate((ss.norm(0,1).rvs((n,2)),ss.norm(1,1).rvs((n,2))),axis=0)
    #generate the same number of 0s and 1s of the size of 1st and second array
    #and join them along axis0
    outcomes = np.concatenate((np.repeat(0,n),np.repeat(1,n)),axis=0)
    return (points,outcomes)


def make_prediction_grid(predictors,outcomes,h,k,limits):
    """note that it's np.arange not arrange
    classify every single point on the prediction grid"""
    (xmin,xmax,ymin,ymax) = limits
    #creates arrays for x's and y's, determines the complexity of the grid
    xs = np.arange(xmin,xmax,h)
    ys = np.arange(ymin,ymax,h)
    #returns two matrices of X's and Y's for grid creation
    #using **meshgrid we get all the needed combinations for plotting the dots for the grid vertices
    xx,yy = np.meshgrid(xs,ys)
    #look up the previous reference bruh
    prediction_grid = np.zeros(xx.shape,dtype = int)
    #eg. list(enumerate(["potato","tomato"])) returns [(0,"potato"),(1,"tomato")]
    for i,x in enumerate(xs):
        for j,y in enumerate(ys):
            #iterate through each point of the grid and classify it.
            p = np.array([x,y])
            prediction_grid[j,i] = knn_predict(p,predictors,outcomes,k)
    return (xx,yy,prediction_grid)


def plot_prediction_grid (xx, yy, prediction_grid):#, filename):
    """you didn't make this one, was imported from the harvard repo,
    Plots KNN predictions for every point on the grid."""
    from matplotlib.colors import ListedColormap
    background_colormap = ListedColormap (["hotpink","lightskyblue", "yellowgreen"])
    observation_colormap = ListedColormap (["red","blue","green"])
    plt.figure(figsize =(10,10))
    plt.pcolormesh(xx, yy, prediction_grid, cmap = background_colormap, alpha = 0.5)
    plt.scatter(predictors[:,0], predictors [:,1], c = outcomes, cmap = observation_colormap, s = 50)
    plt.xlabel('Variable 1'); plt.ylabel('Variable 2')
    plt.xticks(()); plt.yticks(())
    plt.xlim (np.min(xx), np.max(xx))
    plt.ylim (np.min(yy), np.max(yy))
  #  plt.savefig(filename)

"""
#predetermined points

points = np.array([[1,1],[1,2],[1,3],[2,1],[2,2],[2,3],[3,1],[3,2],[3,3]])
#classes/types of points[]
outcomes = np.array([0,0,0,0,0,1,1,1,1,1,1,1,1])
#new user defined point to be classified
p = np.array([2.5,2])
#plotting points[] and p
#[:,0]=all rows and/of column 0,[0,:]/[0] = vice versa
plt.plot(points[:,0],points[:,1],"ro")
plt.plot(p[0],p[1],"bo")
"""


"""
#synth points

n=20
(points,outcomes) = generate_synth_data(n)
plt.figure()
plt.plot(points[:n,0],points[:n,1],"ro",label="mean 0")
plt.plot(points[n:,0],points[n:,1],"bo",label = "mean 1")
plt.legend(loc="best")
p = np.array([0.5,0.7])
a = knn_predict(p,points,outcomes,4)
print("red /" if (a==0) else "blue /",a)
"""


#prediction grid

#predictors are the points to be used for the prediction grid
(predictors, outcomes)= generate_synth_data()
k=5
#define the gridsize -3x to +3x and -3y to +3y with 0.1 step
limits = (-3,4,-3,4)
h=0.1
(xx,yy,prediction_grid)=make_prediction_grid(predictors,outcomes,h,k,limits)
plot_prediction_grid(xx,yy,prediction_grid)


"""
scikitlearn

from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()
predictors = iris.data[:,0:2]
outcomes = iris.target
plt.plot(predictors[outcomes==0][:,0],predictors[outcomes==0][:,1],"ro")
plt.plot(predictors[outcomes==1][:,0],predictors[outcomes==1][:,1],"bo")
plt.plot(predictors[outcomes==2][:,0],predictors[outcomes==2][:,1],"go")
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(predictors,outcomes)
sk_predictions = knn.predict(predictors)
"""


'''
**meshgrid
The purpose of meshgrid is to create a rectangular grid out of an array of x values and an array of y values.

So for example, if we want to create a grid where we have a point at each integer value between 0 and 4 in both the x and y directions. To create a rectangular grid, we need every combination of the x and y points.

This is going to be 25 points, right? So if we wanted to create an x and y array for all of these points, we could do the following.

x[0,0] = 0    y[0,0] = 0
x[0,1] = 0    y[0,1] = 1
x[0,2] = 0    y[0,2] = 2
x[0,3] = 0    y[0,3] = 3
x[0,4] = 0    y[0,4] = 4
x[1,0] = 1    y[1,0] = 0
x[1,1] = 1    y[1,1] = 1
...
x[4,3] = 4    y[4,3] = 3
x[4,4] = 4    y[4,4] = 4

This would result in the following x and y matrices, such that the pairing of the corresponding element in each matrix gives the x and y coordinates of a point in the grid.

x =   0 0 0 0 0        y =   0 1 2 3 4
      1 1 1 1 1              0 1 2 3 4
      2 2 2 2 2              0 1 2 3 4
      3 3 3 3 3              0 1 2 3 4
      4 4 4 4 4              0 1 2 3 4

We can then plot these to verify that they are a grid:

plt.plot(x,y, marker='.', color='k', linestyle='none')
'''