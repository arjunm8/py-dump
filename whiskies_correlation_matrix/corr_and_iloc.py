# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 00:46:53 2017

@author: Arjun
"""

import matplotlib.pyplot as plt
import pandas as pd

whiskey_data = pd.read_csv("whiskies.csv")
#whiskey_data.columns
#we need the data from the columns related to flavours ie 2-13
#iloc = index locations [all rows, column2 - column13(inclusive)]
flavors = whiskey_data.iloc[:,2:14] 
flavors_corr = pd.DataFrame.corr(flavors)
plt.figure(figsize=(10,10))
#pseudocolor
#will be 12x12 matrix with color densities of the correlations of x,y
#ofc correlation with itself is 1.
plt.pcolor(flavors_corr)
plt.colorbar()
print(flavors.columns)
'''
Let's find out about correlations of different flavor attributes.
In other words, we'd like to learn whether whiskies
that score high on, say, sweetness also score high on the honey attribute.
We'll be using the core function to compute correlations
across the columns of a data frame.
There are many different kinds of correlations,
and by default, the function uses what is
called Pearson correlation which estimates
linear correlations in the data.
In other words, if you have measured attributes for two variables,
let's call them x and y the Pearson correlation coefficient
between x and y approaches plus 1 as the points in the xy scatterplot approach
a straight upward line.
But what is the interpretation of a correlation
coefficient in this specific context?
A large positive correlation coefficient indicates
that the two flavor attributes in question
tend to either increase or decrease together.
In other words, if one of them has a high score
we would expect the other, on average, also to have a high score.

Clearly, correlation of any dimension with itself is exactly plus 1
but other strong correlations exist too.
For example, heavy body is associated with smokiness.
In contrast, it seems that a floral flavor is the opposite of full body
or medicinal notes.
We can also look at the correlation among whiskies across flavors.
To do this, we first need to transpose our table.
Since these whiskies are made by different distilleries,
we can also think of this as the correlation
between different distilleries in terms of the flavor profiles of the whiskies
that they produce.
'''