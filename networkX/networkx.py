# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 16:11:58 2017

@author: Arjun
"""

import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import bernoulli

'''
G = nx.Graph()

G.add_node(1)

G.add_nodes_from(["u","v"])

G.nodes()
G.add_edge("u","v")

G.add_edges_from([(1,3),(1,4),(1,6)])

G.edges()
Out[15]: [(1, 2), (1, 3), (1, 4), (1, 6), ('u', 'v')]

G.remove_node(1)
G.remove_edges_from([(1,2),(1,3)])
G.number_of_edges()
G.number_of_nodes()

#network of karateclub members' friendship
G = nx.karate_club_graph()

nx.draw(G,with_labels=True,node_color="lightblue",edge_color="grey")
#degree of each node(expresses number of friends)
G.degree()
#same outputs
G.degree()[33]
G.degree(33)
#returns true
G.degree(0) is G.degree()[0]
'''
#manual ER network, ER = Erdos-Renyi
#p = probability
#create empty graph
#add all n nodes in graph
#loop ove all pairs of nodes
#add an edge with prob p

#returns 0 or 1
#bernoulli.rvs(p=0.2)
'''
def er_graph(nodes,prob):
    "generate an ER graph"
    G = nx.Graph()
    G.add_nodes_from(range(nodes))
    for node1 in G.nodes():
        for node2 in G.nodes():
            if node1<node2 and bernoulli.rvs(p=prob):
                G.add_edge(node1,node2)        
    return G

#nx.draw(er_graph(50,0.08),node_size=40,node_color="gray")

def plot_degree_distribution(G):
    plt.hist(list(G.degree().values()),histtype="step")
    plt.xlabel("Degree $k$")
    plt.xlabel("$p(k)$")
    plt.title("degree distribution")

G1 = er_graph(500,0.08)
plot_degree_distribution(G1)
G2 = er_graph(500,0.08)
plot_degree_distribution(G2)
G3 = er_graph(500,0.08)
plot_degree_distribution(G3)
'''
import numpy as np
a1 = np.loadtxt("adj_allVillageRelationships_vilno_1.csv",delimiter=",")
a2 = np.loadtxt("adj_allVillageRelationships_vilno_2.csv",delimiter=",")
G1 = nx.to_networkx_graph(a1)
G2 = nx.to_networkx_graph(a2)
