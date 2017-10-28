import random

random.seed(1)

# basically imagine a vector of numbers and each number will take the average
#of it's neighbors and itself for a smoother graph 
def moving_window_average(x, n_neighbors=1):
    n = len(x)
    width = n_neighbors*2 + 1
    x = [x[0]]*n_neighbors + x + [x[-1]]*n_neighbors
    means=[]
    for i in range(0,n):
        sum=0
        for j in range(i,i+width):
            sum+=x[j]
        means.append(sum/(width))
    return means


random.seed(1)

x = []
for i in range(1000):
    x.append(random.uniform(0,1))
    
Y = [x]
for j in range(1,10):
    Y.append(moving_window_average(x,j))

ranges = []
for lsts in Y:
    ranges.append(max(lsts)-min(lsts))
print(ranges)