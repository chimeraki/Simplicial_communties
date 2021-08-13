#identifying clique communities of simplicial complexes of arbitrary dimension
#python 3.6
#Sanjukta Krishnagopal s.krishnagopal@ucl.ac.uk
#August 2021

from numpy import *
from matplotlib.pyplot import *
import scipy.linalg as linalg
import scipy
import random
import matplotlib as plt
import math
import networkx as nx
from networkx.generators.classic import empty_graph, path_graph, complete_graph
from scipy.sparse import coo_matrix
import json
import community
import itertools
from functools import reduce
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import eigs
from collections import Counter
from scipy.linalg import null_space
import plotly
from collections import defaultdict
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics.cluster import adjusted_mutual_info_score 

#extract boundary matrix
def incidence(G, verbose=False):
    # G is a networkx graph
    C = nx.find_cliques(G)

    # Sort each clique, make sure it's a tuple
    C = [tuple(sorted(c)) for c in C]

    #  Enumerate all simplices

    # S[k] will hold all k-simplices
    # S[k][s] is the ID of simplex s
    S = []
    for k in range(0, max(len(s) for s in C)):
        # Get all (k+1)-cliques, i.e. k-simplices, from max cliques mc
        Sk = set(c for mc in C for c in itertools.combinations(mc, k + 1))
        # Check that each simplex is in increasing order
        assert (all((list(s) == sorted(s)) for s in Sk))
        # Assign an ID to each simplex, in lexicographic order
        S.append(dict(zip(sorted(Sk), range(0, len(Sk)))))

    # D[k] is the kth boundary operator 
    D = [None for _ in S]
    ind = [None for _ in S]

    # D[0] is the zero matrix
    D[0] = lil_matrix((1, G.number_of_nodes()))

    # Construct D[1], D[2], ...
    for k in range(1, len(S)):
        ind[k-1] = S[k-1].keys()
        D[k] = lil_matrix((len(S[k - 1]), len(S[k])))
        SIGN = np.asmatrix([(-1) ** i for i in range(0, k + 1)]).transpose()

        for (ks, j) in S[k].items():
            # Indices of all (k-1)-subsimplices s of the k-simplex ks
            I = [S[k - 1][s] for s in sorted(itertools.combinations(ks, k))]
            D[k][I, j] = SIGN.squeeze()
            
    #remove a few simplices (generate holes)
    #ideal is removing 25 and 27
    global rem
    a=25 # which triangles to remove. order such that a > b
    b=18
    #c=28
    rem=str(a)#+'_'+str(b) #for saving file name
    S_flip = dict((y,x) for x,y in S[2].items())
    D[2][:,a] = 0 #remove a triangle 
    S[2].pop(S_flip[a])
    #D[2][:,b] = 0
    #S[2].pop(S_flip[b])
    #D[2][:,c] = 0
    #S[2].pop(S_flip[c])
    
    # Check that D[k-1] * D[k] is zero
    assert (all((0 == np.dot(D[k - 1], D[k]).count_nonzero()) for k in range(1, len(D))))

    #  Compute rank and dimension of kernel of the boundary operators
    # Rank and dimker
    rk = [np.linalg.matrix_rank(d.todense()) for d in D]
    ns = [(d.shape[1] - rk[n]) for (n, d) in enumerate(D)]

    # Betti numbers
    # B[0] is the number of connected components
    B = [(n - r) for (n, r) in zip(ns[:-1], rk[1:])]
    print ('Betti numbers are :', B)
    return S, D 


def list_duplicates(seq):
    tally = defaultdict(list)
    for i,item in enumerate(seq):
        tally[item].append(i)
    return ((key,locs) for key,locs in tally.items() 
                            if len(locs)>1)

def supportonly_overlap_check(h_vec):
     subset_support = []
     for i1 in range(len(h_vec)):
         for i2 in range(i1):
             s1 = h_vec[i1]
             s2 = h_vec[i2]
             if s1 != s2:
                 if len(s1)<len(s2):
                     if set(s1).intersection(set(s2)):
                         subset_support.append((i1,i2))
                 if len(s2)<len(s1):
                     if set(s2).intersection(set(s1)):
                         subset_support.append((i2,i1))
     return subset_support

def data(dset, A = None):
    #baboon social interaction
    if dset == 'adjacency':
        assert (any(A)!=None)
        G = nx.from_numpy_matrix(A)


A =np.array([[0,0,0,0,1,1,0,0,0,0,0,0,1],[0,0,0,0,0,1,1,0,1,1,1,1,0],[0,0,0,0,0,0,1,1,0,0,0,0,0],[0,0,0,0,1,0,0,1,0,0,0,0,0],[1,0,0,1,0,1,0,1,0,0,0,0,0],[1,1,0,0,1,0,1,0,1,0,0,0,0],[0,1,1,0,0,1,0,1,0,0,0,0,0],[0,0,1,1,1,0,1,0,0,0,0,0,0],[0,1,0,0,0,1,0,0,0,1,1,0,0],[0,1,0,0,0,0,0,0,1,0,1,1,0],[0,1,0,0,0,0,0,0,1,1,0,1,0],[0,1,0,0,0,0,0,0,0,1,1,0,0],[1,0,0,0,0,0,0,0,0,0,0,0,0]]) #can input an adjacency matrix

G = data('citation', A).to_undirected()
G=nx.karate_club_graph()



#filtration: comment if not thresholding graph
th= np.percentile([w for n1, n2, w in G.edges(data="weight")],50)
G.remove_edges_from([(n1, n2) for n1, n2, w in G.edges(data="weight") if w < th])
edges = G.edges()
nodes = G.nodes()
print ('number of nodes: '+str(len(nodes))+' edges: ' + str(len(edges)))
avg_deg  =sum([val for (node, val) in G.degree()])/float(len(nodes))
print ('average_degree:' + str(avg_deg))
remove = [node for node,degree in dict(G.degree()).items() if degree < avg_deg*3]
G.remove_nodes_from(remove)

edges = G.edges()
nodes = G.nodes()
node_names = [ str(G.nodes[i]["club"][0]) + str(i) for i in range(len(nodes))]

print ('number of nodes: '+str(len(nodes))+' edges: ' + str(len(edges)))
print ('average_degree:' + str(sum([val for (node, val) in G.degree()])/float(len(nodes))))

S, im = incidence(G, verbose=True)  # boundary matrix, incidence matrix
L_up = [None for _ in im]
L_down = [None for _ in im]
L_hodge = [None for _ in im]
eig_val_u = [None for _ in im]
eig_vec_u = [None for _ in im]
eig_val_d = [None for _ in im]
eig_vec_d = [None for _ in im]
ker_Ldown =  [None for _ in im]
ker_Lup =  [None for _ in im]
sync =  [None for _ in im]
unique_data =  [None for _ in im]

                                    
for i in range(len(im)-1):    #iterate over ith simplicial complexes (0 - nodes, 1 - lines etc)
     L_up[i] = np.dot(im[i+1],im[i+1].T)
     L_down[i] = np.dot(im[i].T,im[i])
     L_hodge[i] = L_up[i] + L_down[i]
    
     #compute eigenvectors for Lup
     val_u,vec_u = np.linalg.eig(L_up[i].todense()) #this implementation tends to find sparse eigenvectors
     idx = val_u.argsort()[::]   
     eig_val_u[i], eig_vec_u[i] = val_u[idx],vec_u.T[idx] #note that the eigenvectors are column vectors and hence we take transpose
     pos_nonzero_u = next((index for index,value in enumerate(val_u[idx]) if value > 0.000001), None)
     
     #use nonzero eigenvector to get up-clique communities
     if pos_nonzero_u == len(eig_val_u[i])-1:
         print ('no nonzero eigenvalues in Lup')
     nonzero_eigval = eig_val_u[i][pos_nonzero_u:]
     nonzero_eigvec = eig_vec_u[i][pos_nonzero_u:]
     print ('nonzero eigenvalues: '+str(nonzero_eigval))
     if len(np.unique(np.around(nonzero_eigval,5)))!=len(nonzero_eigval):
         print ('*******Degeneracy in eigenvalues')

     #identifying support
     supprt = []
     for j in range(len(nonzero_eigval)):
         supprt.append(list(np.where(abs(nonzero_eigvec[j])>0.000001)[1]))
         #rotate the eigenvectors such that support is independent

         
     #take union of support of eigenvectors with intersecting support
     print ('naive support: '+str( supprt))
     supprt = np.array(supprt)

     #consider all eigenvectors
     
     #degenerate
     deg = list_duplicates(np.around(nonzero_eigval,5))
     deg_vals=[]
     for d in deg:
         deg_vals.append(np.round(d[0],5))
         union_deg = reduce(set.intersection, [set(x) for x in supprt[d[1]]])
         if union_deg:
             for v in d[1]:
                 supprt[v] = list(reduce(set.union, [set(x) for x in supprt[d[1]]]))

     #nondegenerate
     nondeg = [list(nonzero_eigval).index(i) for i in nonzero_eigval if np.round(i,5) not in list(deg_vals)]
     supprt = np.unique(supprt)

                

     #identify nodes in clique communities
     S_flip =  dict((y,x) for x,y in S[i].items())
     clique_comm = []
     for j in range(len(supprt)):
         node_pairs = [S_flip[i] for i in supprt[j]]
         if i ==0:
             cc_nodes = supprt[j]
         else:
             cc_nodes  = np.unique(list(itertools.chain(*[S_flip[i] for i in supprt[j]])))
         if len(cc_nodes) > i+1:
             clique_comm.append(list(cc_nodes))
     unique_data[i] = [list(x) for x in set(tuple(x) for x in clique_comm)]
     print ('number of k=' +str(i)+' up clique communities: '+ str(len(unique_data[i])))
     print ('Nodes in up clique communities are ' +str(unique_data[i]))

   

#Adjusted mutual information: extract all community affiliations of each node

clubs = [G.nodes[j]["club"] for j in range(len(nodes))]
un_cl = np.array(np.unique(clubs))
cluster=[]
clb = []
for j in range(len(nodes)):
    cluster.append([])
    clb.append(int(np.where(un_cl==G.nodes[j]["club"])[0]))
    for ud in range(len(unique_data[1])):        
        if j in unique_data[1][ud]:
            cluster[-1].append(ud)
    if not cluster[-1]:
        cluster[-1] = [-1]

AMI = []
for e in range(100):
    print (e)
    f = []
    for j in cluster:
        f.append(random.choice(j))
    AMI.append(adjusted_mutual_info_score(f,clb))
        


    
    

#plotting using plotly
node_labels = node_names #list(G.nodes())
#pos = nx.spring_layout(G)
pos={0: array([-0.34499372,  0.03806757]),

style_e = ["dash", "dot", "longdash", "dashdot", "longdashdot"] #to mark tetrahedron connected by triangles
colors_e = px.colors.qualitative.Bold
edge_x = []
edge_y = []
edge_color = []
edge_style = [] 
for edge in G.edges():
    a,b = list(edge) 
    x0, y0 = pos[a][0], pos[a][1]
    x1, y1 = pos[b][0], pos[b][1]
    '''for aff in range(len(unique_data[0])):
        if set(np.sort([a,b])).issubset(set(unique_data[0][aff])):
            edge_color.append(colors_e[aff])
        else:
            edge_color.append('#888')
    for aff in range(len(unique_data[2])):
        if set(np.sort([a,b])).issubset(set(unique_data[2][aff])):
            edge_style.append(style_e[aff])
        else:
            edge_style.append('solid')'''
    edge_x.append(x0)
    edge_x.append(x1)
    edge_x.append(None)
    edge_y.append(y0)
    edge_y.append(y1)
    edge_y.append(None)

edges_list=[ dict(type='scatter',
             x=edge_x,
             y=edge_y,
             mode='lines',
                  line=dict(width=1, color='black'))]
'''edges_list=[ dict(type='scatter',
             x=edge_x[3*k:3*k+2],
             y=edge_y[3*k:3*k+2],
             mode='lines',
             line=dict(width=3, dash = edge_style[k], color=edge_color[k]))  for k, e in enumerate(edges)]'''



node_x = []
node_y = []
for node in G.nodes():
    x, y = pos[node]
    node_x.append(x)
    node_y.append(y)

node=dict(type='scatter',
           x=node_x,
           y=node_y,
           mode='markers',
           opacity=0.4,
           marker=dict(size=10, color='blue'))

annotations = []
annotations.append(dict(x = node_x,
                        y=node_y,
                        text = node_labels,
font=dict(color='black', size=10)))

axis = dict(showline=False,
          zeroline=False,
          showgrid=False,
          showticklabels=False,
          title='' 
          )
layout = plotly.graph_objs.Layout(
    showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    xaxis=plotly.graph_objs.XAxis(axis),
    yaxis=plotly.graph_objs.YAxis(axis),
    )
mapping = dict(zip(G, np.arange(len(nodes))))
map_flip =  dict((y,x) for x,y in mapping.items())

colors = px.colors.qualitative.Light24
figure()
labels = [str(k) for k in range(len(pos))]
#get the triangles
triangles = list()
triangle_nodes = []
for i in nodes:
    neighbors = [n for n in G.neighbors(i)]
    for j,k in itertools.combinations(neighbors,2):
        if G.has_edge(j,k):
            p = list(sort([i,j,k]))
            if p in triangle_nodes or tuple(p) not in S[2].keys():
                continue
            else:
                for affil in range(len(unique_data[1])): #coloring triangles by their induced community affiliation based on the k-simplicial communities denoted by unique_data[k]
                    if set(p).issubset(set(unique_data[1][affil])):
                        aff = affil
                triangle_nodes.append(list(sort([i,j,k])))
                triangles.append(dict(type='path',
                          path='M {0} {1} L {2} {3} L {4} {5} Z'.format(
                              pos[i][0],
                              pos[i][1],
                              pos[j][0],
                              pos[j][1],
                              pos[k][0],
                              pos[k][1]),
                          fillcolor=colors[aff],
                                      opacity=0.2,
                                      line=dict(color='grey')
                    ))
                

layout.update(shapes=triangles)
fig = plotly.graph_objs.Figure(data=edges_list+[node], layout=layout)
fig.update_annotations(font_size = 6, visible = True)
fig.update_traces(textposition="top left")
fig.update_layout( xaxis_range=[-1.1, 1.1] )
for i in range(len(node_x)):
    fig.add_annotation(dict(x=node_x[i], y= node_y[i], text = node_labels[i],showarrow=False),font=dict(color='black', size=16))
#fig = plotly.graph_objs.Figure(data=[edges_trace,node_trace], layout=layout)
fig.update_traces(marker_showscale=False) #remove colorbar
fig.write_image('karate_'+str(rem)+'_MImean_%0.3f_var_%0.3f_.pdf' %(np.mean(MI),np.std(MI)))
#plotly.offline.plot(fig)
