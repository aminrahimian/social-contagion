import networkx as NX
from models import two_d_lattice_union_Erdos_Renyi, two_d_lattice_union_diagnostics
import matplotlib.pyplot as plt
import math
plt.rc('text', usetex=True)
plt.rc('font', family='serif')



n = 25
root_n = int(math.sqrt(n))

# plot the 2 dimensional lattice graph with ER with different color for ER
plt.figure(1,(6,6))
G = two_d_lattice_union_Erdos_Renyi(n, c=4, seed=None, color_the_edges=True, weight_the_edges=True)
colors = [G[u][v]['color'] for u,v in G.edges]
weights = [G[u][v]['weight'] for u,v in G.edges]
pos = {}
for i in range(root_n):
    for j in range(root_n):
        pos[(i * root_n) + j] = (i, j)
plt.axis('off')
NX.draw(G, pos=pos, edge_color=colors, width=weights)
plt.savefig('./figures/lattice_erdos(color).pdf')

# plot the 2 dimensional lattice graph with ER
plt.figure(2,(6,6))
G = two_d_lattice_union_Erdos_Renyi(n, c=4, seed=None, weight_the_edges=True)
# colors = [G[u][v]['color'] for u,v in G.edges]
weights = [G[u][v]['weight'] for u,v in G.edges]
pos = {}
for i in range(root_n):
    for j in range(root_n):
        pos[(i * root_n) + j] = (i, j)
plt.axis('off')
NX.draw(G, pos=pos, width=weights)
plt.savefig('./figures/lattice_erdos.pdf')

# plot the 2 dimensional lattice graph with diagnostics
plt.figure(3,(6,6))
G = two_d_lattice_union_diagnostics(n, seed=None, color_the_edges=True, weight_the_edges=True)
colors = [G[u][v]['color'] for u,v in G.edges]
weights = [G[u][v]['weight'] for u,v in G.edges]
pos = {}
for i in range(root_n):
    for j in range(root_n):
        pos[(i * root_n) + j] = (i, j)
plt.axis('off')
NX.draw(G, pos=pos, edge_color=colors, width=weights)
plt.savefig('./figures/lattice_diagnostics.pdf')

