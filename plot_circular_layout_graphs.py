import networkx as NX
import math
import matplotlib.pyplot as plt
import random as RD
from models import cycle_union_Erdos_Renyi

import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

plt.figure(1)
G = cycle_union_Erdos_Renyi(10, k=2, c=1, seed=None, color_the_edges=True)
colors = [G[u][v]['color'] for u,v in G.edges]
weights = [G[u][v]['weight'] for u,v in G.edges]
NX.draw_circular(G, edge_color=colors, width=weights)
plt.text(-0.25,0,'$\mathcal{C}_{1} \\cup \mathcal{G}_{n,2/n}$', fontsize=30,
         fontweight='bold', fontdict=None, withdash=False)

plt.show()


plt.figure(2)

G = cycle_union_Erdos_Renyi(10, k=4, c=0, seed=None, color_the_edges=True)
# G = NX.connected_watts_strogatz_graph(params['size'], params['nearest_neighbors'], params['rewiring_probability'])
colors = [G[u][v]['color'] for u,v in G.edges]
weights = [G[u][v]['weight'] for u,v in G.edges]
NX.draw_circular(G, edge_color=colors, width=weights)
plt.text(-0.05,0,'$\mathcal{C}_{2}$', fontsize=30,
         fontweight='bold', fontdict=None, withdash=False)

plt.show()


plt.figure(3)

G = cycle_union_Erdos_Renyi(10, k=6, c=0, seed=None, color_the_edges=True)
G.add_edge(1,5,color='b')
colors = [G[u][v]['color'] for u,v in G.edges]
NX.draw_circular(G, edge_color=colors, width=weights)
plt.text(-0.25,0,'$\mathcal{C}_{3} \\cup \mathcal{G}_{n,1/n}$', fontsize=30,
         fontweight='bold', fontdict=None, withdash=False)

plt.show()

