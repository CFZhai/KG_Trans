#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 14:33:07 2022

@author: chaofanzhai
"""

import os 
import networkx as nx
import matplotlib.pyplot as plt
import pandas  as pd 
import random 
random.seed(123)

os.chdir("/Users/chaofanzhai/UMN/Project_memory/data/find_dynamic_relationship")
df = pd.read_csv('word_pair_2.csv')

# check distribution of edge weight first and determine the threshold 
df.describe()


###
# we need get general sense of our graph first 
# so we need visualize a subgraph first
###

## Some key threshold :
##supp1 #同时记住的次数/同时出现的次数
##supp2 #同时记住的次数/记录长度
##cond1 #同时记住的次数/ X记住的次数
##lift2 #同时记住的次数/同时出现的次数 ）/（Y记住的次数/Y出现的次数）

supp_thres1 = df.supp1.quantile(0.9)
supp_thres2 = df.supp2.quantile(0.8)
cond_thres1 =df.cond1.quantile(0.9)
cond_thres2=0
lift_thres1 = 0 
lift_thres2 =df.lift2.quantile(0.9)
sample_size=4
hop_n=1

def get_subgraph(df,supp_thres1,supp_thres2,cond_thres1,cond_thres2,lift_thres1,lift_thres2,sample_size,hop_n):
    import random 
    random.seed(123)
    
    df_temp = df[(df.supp1>supp_thres1) &(df.cond1>cond_thres1) &(df.lift1>lift_thres1) \
                 & (df.supp2>supp_thres2) &(df.cond2>cond_thres2) &(df.lift2>lift_thres2) ]

    ##tranform it to a graph 

    G= nx.from_pandas_edgelist(df = df_temp, source = 'a', target = 'b', edge_attr=['supp1', 'cond1', 'lift1', 'supp2', 'cond2', 'lift2'])

    #check if it is a connected graph 
    print("is it a connected graph ",nx.is_connected(G))
    # true,,it will be a good news.
    #False , bad news 

    #sample sample_size source nodes

    nodes_set = list(set(df_temp.a).union(set(df_temp.b)) )
    
    print( "# of nodes in graph:",G.number_of_nodes())
    
    sample_source_nodes = random.sample(nodes_set, sample_size)
    
    
    
    #Use BFS to get all connected nodes from these 3 sources nodes.
    # we can difine the path depth,1 2 ,3,4 hop..

    sample_edges=[]
    # n-hop depth
    for i in sample_source_nodes:
        related_nodes= list(nx.bfs_edges(G, source=i, depth_limit=hop_n)) #
        sample_edges.extend(related_nodes)
    
    sample_nodes_set=set()
    for i in sample_edges:
        sample_nodes_set= sample_nodes_set.union(set(i))
    sample_nodes=   list(sample_nodes_set)  
    

    #build the subgraph 
    H = G.subgraph(sample_nodes)
    
    print("# of nodes in subgrpah:",H.number_of_nodes())
    return df_temp,G, sample_source_nodes, H

df_temp,G, sample_source_nodes,H= get_subgraph(df,supp_thres1,supp_thres2,cond_thres1,cond_thres2,lift_thres1,lift_thres2,sample_size,hop_n)


## plot sampled graph 
plt.figure(figsize=(40,30))
nx.draw(H,pos=nx.spring_layout(H) ,with_labels=True,alpha=0.9,node_size=10,width=0.08, font_size=10)
plt.show()



## summary stats for the filtered graph ?
nx.eigenvector_centrality(G)

####################################
# cluster ,community finding 
#ref : https://graphsandnetworks.com/community-detection-using-networkx/
##############################
from networkx.algorithms.community import girvan_newman,greedy_modularity_communities

def set_node_community(G, communities):
    '''Add community to node attributes'''
    for c, v_c in enumerate(communities):
        for v in v_c:
            # Add 1 to save 0 for external edges
            G.nodes[v]['community'] = c + 1
def set_edge_community(G):
    '''Find internal edges and add their community to their attributes'''
    for v, w, in G.edges:
        if G.nodes[v]['community'] == G.nodes[w]['community']:
            # Internal edge, mark with community
            G.edges[v, w]['community'] = G.nodes[v]['community']
        else:
            # External edge, mark as 0
            G.edges[v, w]['community'] = 0
def get_color(i, r_off=1, g_off=1, b_off=1):
    '''Assign a color to a vertex.'''
    r0, g0, b0 = 0, 0, 0
    n = 16
    low, high = 0.1, 0.9
    span = high - low
    r = low + span * (((i + r_off) * 3) % n) / (n - 1)
    g = low + span * (((i + g_off) * 5) % n) / (n - 1)
    b = low + span * (((i + b_off) * 7) % n) / (n - 1)
    return (r, g, b)

communities = greedy_modularity_communities(H)
print("# of communities:",len(communities))

 # Set node and edge communities
set_node_community(H, communities)
set_edge_community(H)
node_color = [get_color(H.nodes[v]['community']) for v in H.nodes]
# Set community color for edges between members of the same community (internal) and intra-community edges (external)
external = [(v, w) for v, w in H.edges if H.edges[v, w]['community'] == 0]
internal = [(v, w) for v, w in H.edges if H.edges[v, w]['community'] > 0]
internal_color = ['black' for e in internal]
    

H_pos = nx.spring_layout(H)
plt.figure(figsize=(40,30))
plt.rcParams.update({'figure.figsize': (40, 30)})
plt.style.use('default')
# Draw external edges
nx.draw_networkx(
    H,
    pos=H_pos,
    node_size=0,
    edgelist=external,
    edge_color="silver")
# Draw nodes and internal edges
nx.draw_networkx(
    H,
    pos=H_pos,
    node_color=node_color,
    edgelist=internal,
    edge_color=internal_color)




## ## ## ## ## ## ## ## ## ## 
## plot G communities
## ## ## ## ## ## ## ## ## ## ## 
plt.figure(figsize=(40,30))
pos = nx.spring_layout(G, k=0.1)
plt.rcParams.update({'figure.figsize': (40,30)})
nx.draw_networkx(
     G,
     pos=pos,
     node_size=0,
     edge_color="#444444",
     alpha=0.05,
     with_labels=False)


communities = sorted(greedy_modularity_communities(G), key=len, reverse=True)
len(communities)

plt.figure(figsize=(40,30))
plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams.update({'figure.figsize':(40,30)})
plt.style.use('dark_background')
# Set node and edge communities
set_node_community(G, communities)
set_edge_community(G)
# Set community color for internal edges
external = [(v, w) for v, w in G.edges if G.edges[v, w]['community'] == 0]
internal = [(v, w) for v, w in G.edges if G.edges[v, w]['community'] > 0]
internal_color = ["black" for e in internal]
node_color = [get_color(G.nodes[v]['community']) for v in G.nodes]
# external edges
nx.draw_networkx(
    G,
    pos=pos,
    node_size=0,
    edgelist=external,
    edge_color="silver",
    node_color=node_color,
    alpha=0.2,
    with_labels=False)
# internal edges
nx.draw_networkx(
    G, pos=pos,
    edgelist=internal,
    edge_color=internal_color,
    node_color=node_color,
    alpha=0.05,
        with_labels=False)


###  heterogenous 
## dict{node: type }

# sparsity 







